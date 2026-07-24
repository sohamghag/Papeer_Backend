from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph,add_messages,START,END
from langchain_core.messages import BaseMessage,AIMessage,HumanMessage,ToolMessage,SystemMessage
from typing import TypedDict, Annotated,Literal
from pydantic import Field
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
from pydantic import BaseModel
import os 
from contextlib import asynccontextmanager
from langgraph.prebuilt import (
    ToolNode,
    InjectedState, tools_condition
)
from langchain_core.tools import tool,InjectedToolCallId
from vector_store import search, MissingApiKeyError
# from tavily import TavilyClient
from tavily import AsyncTavilyClient
load_dotenv()
api_keys_tav=os.getenv("TAV")
max_retrieval_attempt = 3

def get_llm(state: dict) -> ChatOpenAI:
    """Builds the OpenAI client for this invocation — requires the user's own key."""
    api_key = state.get("openai_api_key")
    if not api_key:
        raise MissingApiKeyError("An OpenAI API key is required.")
    return ChatOpenAI(model="gpt-4.1-mini", api_key=api_key)

def get_kimi_llm(state: dict) -> ChatOpenAI:
    """Builds the Kimi client for this invocation — requires the user's own key."""
    api_key = state.get("kimi_api_key")
    if not api_key:
        raise MissingApiKeyError("A Kimi API key is required.")
    return ChatOpenAI(
        api_key=api_key,
        base_url="https://api.moonshot.ai/v1",
        model="moonshot-v1-32k",
    )

def merge_docs(existing, new):
    if new == []:
        return []
    existing = existing or []
    new = new or []
    seen = {hash(d.page_content[:100]) for d in existing}
    deduped = [d for d in new if hash(d.page_content[:100]) not in seen]
    return existing + deduped

class RAGState(TypedDict):
    session_id:str  # needed while performing retrieval -->session_id is important to fetch the exact collection for store
    query:str | None
    route:str | None
    messages:Annotated[list[BaseMessage],add_messages]
    retrieved_docs: Annotated[list[Document], merge_docs]
    retrieval_attempts: int
    is_relevant:bool | None
    rewrite_query_count:int | None
    relevancy_reason:str | None
    claim_verdict: str | None
    superseding_papers: list[dict] | None
    answer:str |None
    openai_api_key: str | None  # required — no fallback to the app's own key
    kimi_api_key: str | None    # required — no fallback to the app's own key

class RouterDecision(BaseModel):
    route_decision : Literal[
        "direct_answer",
        "retrieve",
        "verify_claim"
    ]

class RetrieverInput(BaseModel):
    query: str

class WebSearchInput(BaseModel):
    query: str

class RelevancyDecision(BaseModel):
    is_relevant: bool
    relevancy_reason: str

class RewrittenQuery(BaseModel):
    rewritten_query:str

class SupersedingPaper(BaseModel):
    title: str
    url: str
    summary: str

class ClaimVerificationResult(BaseModel):
    verdict_summary: str
    superseding_papers: list[SupersedingPaper]

# All Decision Functions
def route_decision(state: RAGState)-> Literal["verify_claim","generate_answer","agent_node"]:
    print(state["route"])
    if state["route"] == "verify_claim":
        return "verify_claim"
    elif state['route'] == "direct_answer":
        return "generate_answer"
    else:
        return "agent_node"
    
def tool_relevancy_decision(state: RAGState) ->Literal["tool_node","relevancy_check","generate_answer"]:
    tc = tools_condition(state) # looks at the last message in state and checks if it has tool_calls:
    if tc == "tools":
        return "tool_node"
    else:
        return "relevancy_check"

def check_relevancy(state: RAGState) -> Literal["generate_answer", "rewrite_query"]:
    if state.get("is_relevant"):
        return "generate_answer"
    if state.get("rewrite_query_count", 0) < 1:
        return "rewrite_query"
    return "generate_answer"  # already rewrote once, give up, answer anyway

# Tools 
@tool(args_schema=RetrieverInput)
async def retrieve_from_vectorstore(query: str, tool_call_id: Annotated[str, InjectedToolCallId],
    session_id: Annotated[str, InjectedState("session_id")],
    openai_api_key: Annotated[str | None, InjectedState("openai_api_key")] = None,
    kimi_api_key: Annotated[str | None, InjectedState("kimi_api_key")] = None):
    """Search the uploaded research paper vector store for relevant passages."""
    print(f"\n>>> ENTERED retrieve_from_vectorstore(query={query!r})")
    try:
        docs = await search(query, session_id, openai_api_key=openai_api_key, kimi_api_key=kimi_api_key)
        if not docs:
            return "No relevant documents found."
        return [
        ToolMessage(content=f"Retrieved {len(docs)} chunk(s).", tool_call_id=tool_call_id), ## this will automatically goes into the message state
        Command(update={"retrieved_docs": docs}),
        ]
    except Exception as e:
        print(f"[retrieve_from_vectorstore] ERROR: {type(e).__name__}: {e}")
        raise

@tool(args_schema=WebSearchInput)
async def web_search(query:str,tool_call_id: Annotated[str, InjectedToolCallId]):
    """Search the web for current or supplementary information using Tavily."""
    print(f"\n>>> ENTERED web_search(query={query!r})")
    try:
        client = AsyncTavilyClient(api_key=api_keys_tav)
        results = await client.search(query, max_results=5)
        if not results.get("results"):
            return "No web results found."

        web_docs = [
        Document(
            page_content=r["content"],
            metadata={"url": r["url"], "title": r.get("title", "Web Result"), "source_type": "web"},  
        )
        for r in results["results"]
        ]
        summary = f"Found {len(web_docs)} web result(s) for: {query}"
        return [
            ToolMessage(content=summary, tool_call_id=tool_call_id),
            Command(update={"retrieved_docs":  web_docs}),
        ]
        
    except Exception as e:
        print(f"[web_search] ERROR: {type(e).__name__}: {e}")
        raise

tools = [retrieve_from_vectorstore,web_search]

# AIMessage form Agent_Node goes to tool_node Tool node calls the function
tool_node = ToolNode(tools)

async def router(state: RAGState):
    # router will decide the route and store in the state 
    query = state["messages"][-1].content
    prompt = ChatPromptTemplate([
    (
        "system",
        """
        You are a routing assistant for a research paper Q&A system.

Classify the user's query into EXACTLY ONE of the following routes:

1. retrieve
2. verify_claim
3. direct_answer



-----------------------------------
ROUTE: retrieve
-----------------------------------

Use retrieve when:

A. The user is asking about the content of uploaded papers, reports, PDFs, articles, documents, or research material.

Examples:
- What does the paper conclude?
- Summarize the report.
- What methodology was used?
- Who are the authors?
- What are the results?
- What does the document say about sustainability?
- As per the report, what is sustainable development?
- According to the paper, what is the main finding?
- Based on the uploaded PDF, explain the introduction.
- What is written in the report regarding climate change?

IMPORTANT:

If the user mentions ANY of the following, ALWAYS choose retrieve:

- report
- paper
- document
- pdf
- article
- study
- research
- uploaded file
- uploaded document
- uploaded paper
- uploaded report

and phrases such as:

- as per the report
- according to the report
- according to the paper
- according to the document
- in the report
- in the paper
- from the report
- from the paper
- what does the report say
- what does the paper say
- based on the report
- based on the paper
- based on the uploaded file

Even if the question could be answered from general knowledge, choose retrieve whenever the user appears to be referring to an uploaded document.

B. Questions requiring current, live, or recently-changing information —
   anything whose correct answer could be outdated if answered purely
   from training knowledge.

Examples:
- What is the weather in Mumbai?
- What is the current gold price?
- Who is the current president?
- Latest AI news
- Current stock price of Tesla
- Today's temperature in Delhi
- Who won the most recent FIFA World Cup?
- Who won the last Nobel Prize in Physics?
- What is the latest iPhone model?
- Who is the current CEO of OpenAI?
- What was the result of the most recent election in India?

IMPORTANT: Any question about a "winner," "current holder," "latest,"
"most recent," or "who is now ___" should be treated as requiring
current information — even if it superficially resembles a
general-knowledge "who/what" question like those in direct_answer.

-----------------------------------
ROUTE: verify_claim
-----------------------------------

Use verify_claim when the user wants to determine whether a claim, result, finding, conclusion, or statement from a paper is still valid or has been superseded by newer research.

Examples:
- Is this claim still accurate?
- Has this paper been superseded?
- Is this finding outdated?
- Check whether this conclusion still holds.
- Verify this research claim.

-----------------------------------
ROUTE: direct_answer
-----------------------------------

Use direct_answer ONLY for stable general knowledge questions that do not require retrieval from uploaded documents and do not require current information.

Examples:
- What is softmax?
- Explain backpropagation.
- What is gradient descent?
- Who invented the transformer architecture?
- Explain reinforcement learning.

-----------------------------------
DECISION RULE
-----------------------------------

When in doubt between retrieve and direct_answer,
ALWAYS choose retrieve.

Return ONLY the route field.
        """
    ),
    ("human", "{query}"),
])
    kimi_router = get_kimi_llm(state).with_structured_output(RouterDecision)
    chain = prompt | kimi_router    
    response=await chain.ainvoke({"query":query})

    print("Router_Response",response,route_decision)
    return {
        "route": response.route_decision,
        "query": query,
        "retrieved_docs": [],          
        "retrieval_attempts": 0,       # no reducer → plain overwrite
        "is_relevant": None,           # no reducer → plain overwrite
        "rewrite_query_count": 0,      # no reducer → plain overwrite
    }

def agent_node(state:RAGState):
    # agent_prompt = (
    #     "You are a research assistant gathering context to answer a user's question.\n\n"
    #     "You have two tools available:\n\n"
    #     "1. retrieve_from_vectorstore — searches the uploaded document collection.\n"
    #     "   Use this for ANY question about:\n"
    #     "   - the uploaded paper, book, report, document, or PDF\n"
    #     "   - its content, conclusion, summary, authors, methodology, findings\n"
    #     "   - anything the user is asking 'from' or 'about' the document\n"
    #     "   - query: phrase it to best match relevant chunks\n"
    #     "2. web_search — searches the live web via Tavily.\n"
    #     "   Use this ONLY when the question is explicitly about:\n"
    #     "   - current events, live data, weather, stock prices, latest news\n"
    #     "   - something clearly NOT in any uploaded document\n"
    #     "   - optimized_query: concise keyword-rich search query\n"
    #     "   - max_results: 1–10\n\n"
    #     "STRICT RULES:\n"
    #         "- If the user asks about the uploaded document/paper/book/reports → ONLY call retrieve_from_vectorstore. NEVER call web_search.\n"
    #         "- web_search is ONLY for live data (weather, prices, news) — NOT for document questions.\n"
    #         "- Call ONE tool total. Stop after one tool call.\n"
    #         "- Do NOT call web_search after retrieve_from_vectorstore or vice versa.\n"
    #     "IMPORTANT RULES:\n"
    #     "- DEFAULT to retrieve_from_vectorstore when in doubt.\n"
    #     "- If the user says 'the book', 'the paper', 'the report', 'the document', "
    #     "'the conclusion', 'the author', 'the findings' → ALWAYS use retrieve_from_vectorstore.\n"
    #     "- NEVER use web_search to answer questions about uploaded documents.\n"
    #     "- Call only one tool per turn.\n"
    #     "- Do NOT produce a final answer. Only call tools to collect context."
    # )
    agent_prompt = (
    "You are a research assistant gathering context to answer a user's question.\n\n"
    "You have two tools available:\n\n"
    "1. retrieve_from_vectorstore — searches the uploaded document collection.\n"
    "   Use this for ANY question about:\n"
    "   - the uploaded paper, book, report, document, or PDF\n"
    "   - its content, conclusion, summary, authors, methodology, findings\n"
    "   - anything the user is asking 'from' or 'about' the document\n"
    "2. web_search — searches the live web via Tavily.\n"
    "   Use this ONLY when the question is explicitly about:\n"
    "   - current events, live data, weather, stock prices, latest news\n"
    "   - something clearly NOT in any uploaded document\n\n"
    "STRICT RULES:\n"
    "- If the user asks about the uploaded document/paper/book/report → ONLY call retrieve_from_vectorstore, NEVER web_search.\n"
    "- web_search is ONLY for live data — NOT for document questions.\n"
    "- DEFAULT to retrieve_from_vectorstore when in doubt.\n\n"
    "CALL LIMIT — READ CAREFULLY:\n"
    "- Call AT MOST ONE tool per turn, and AT MOST ONE tool call total for this question.\n"
    "- After you receive tool results in the conversation, DO NOT call another tool, "
    "even if the results seem incomplete. A separate relevancy check will decide "
    "if the results are good enough, and you will get a chance to search again "
    "with a rephrased query if needed.\n"
    "- If you already have a ToolMessage in the conversation for this question, "
    "respond with plain text acknowledging you have gathered context — do NOT call a tool again.\n"
    "- Do NOT produce the final answer to the user's question yourself — that is handled separately. "
    "Just confirm you have context or call one tool to get it."
    )
    current_attempts = state.get("retrieval_attempts", 0)
    request_llm = get_llm(state)
    # This Fallback is also very important why --> llm can make tool calls as much as it want and we will stuck in the infinite loop between agent_node and tool_node to avoid
    llm_with_brain = request_llm if current_attempts >=max_retrieval_attempt else request_llm.bind_tools(tools, parallel_tool_calls=False)

    # trim to current turn only — find the last HumanMessage, keep from there onward
    all_messages = state["messages"]
    last_human_idx = None
    for idx, msg in enumerate(all_messages):
        if isinstance(msg, HumanMessage):
            last_human_idx = idx

    current_turn_messages = all_messages[last_human_idx:] if last_human_idx is not None else all_messages
    messages = [SystemMessage(content=agent_prompt)] + current_turn_messages

    response = llm_with_brain.invoke(messages)
    updates = {"messages": [response]}
    if getattr(response, "tool_calls", None):
        updates["retrieval_attempts"] = state.get("retrieval_attempts", 0) + 1
    return updates

def relevancy_check(state: RAGState):
    try:
        relevancy_prompt = (
        "You are evaluating whether retrieved chunks can help answer a user's question.\n\n"
        "Be VERY lenient. Return is_relevant=true if:\n"
        "- The chunks contain ANY related information even with different terminology\n"
        "- The topic is similar even if exact words don't match\n"
        "- Partial answers count as relevant\n\n"
        "Return is_relevant=false ONLY if chunks are completely unrelated topics.\n"
        "When in doubt → return true."
        )
        query = state["query"]
        retrieved_docs=state["retrieved_docs"]   
        doc_snippets = "\n\n---\n\n".join(doc.page_content[:600] for doc in retrieved_docs[:3])
        if not doc_snippets:
            return {"is_relevant": False, "relevancy_reason": "No documents were retrieved for the query."}
        relevancy_check_llm=get_kimi_llm(state).with_structured_output(RelevancyDecision)
        response=relevancy_check_llm.invoke([
        {"role": "system", "content": relevancy_prompt},
        {"role": "user", "content": f"Question: {query}\n\nRetrieved chunks:\n{doc_snippets}"}
        ])

        return {"is_relevant":response.is_relevant , "relevancy_reason": response.relevancy_reason}
    except Exception as e:
        print(f"[relevancy_check] ERROR: {type(e).__name__}: {e}")
        raise

def rewrite_query(state: RAGState):
    try:
        rewrite_query_prompt = (
        "You are a query rewriting assistant for a document retrieval system.\n"
        "The previous query failed to find relevant chunks in the uploaded document.\n\n"
        "YOUR ONLY JOB is to rephrase the query using alternative words that mean the SAME thing.\n\n"
        "STRICT RULES:\n"
        "- Keep the same meaning and intent as the original query.\n"
        "- Only change terminology/synonyms, not the topic.\n"
        "- Do NOT broaden, generalize, or add new concepts.\n"
        "- Keep it short — under 10 words.\n\n"
        "Return ONLY the rewritten query. No explanation."
        )

        query=state["query"]
        rewritten_query_llm = get_kimi_llm(state).with_structured_output(RewrittenQuery)
        response = rewritten_query_llm.invoke([
        {"role": "system", "content": rewrite_query_prompt},
        {"role": "user", "content": f"Original query: {query}\n\nWrite an improved search query."}
        ])
        rewritten = response.rewritten_query 

        return {
        "query": rewritten,
        "messages": [HumanMessage(content=rewritten)],
        "retrieved_docs": [],
        "retrieval_attempts": 0,
        "is_relevant": None,
        "rewrite_query_count": state.get("rewrite_query_count", 0) + 1,
        }
    except Exception as e:
        print(f"[rewrite_query] ERROR: {type(e).__name__}: {e}")
        raise

async def verify_claim(state: RAGState):
    try:
        query = state["query"]
        client = AsyncTavilyClient(api_key=api_keys_tav)
        results =await client.search(query, max_results=3)

        lines = []
        for r in results.get("results", []):
            lines.append(f"Title: {r.get('title', '')}\nURL: {r['url']}\nSnippet: {r.get('content', '')[:300]}\n")
        context = "\n\n".join(lines)

        verify_prompt = (
        "You are a research fact-checker. Given a claim and web search results, determine:\n"
        "1. Has this claim been superseded, challenged, or updated by more recent work?\n"
        "2. Identify up to 3 sources from the results that support your verdict.\n\n"
        "Rules:\n"
        "- Use ONLY titles and URLs that appear verbatim in the search results.\n"
        "- For each superseding paper, write a 'summary' field: one sentence explaining "
        "how that specific paper supersedes or challenges the claim.\n"
        "- If the claim still holds, set superseding_papers to an empty list.\n"
        "- verdict_summary should be 1-2 sentences suitable for display to the user."
        )

        verification_llm = get_kimi_llm(state).with_structured_output(ClaimVerificationResult)
        response = await verification_llm.ainvoke([
            {"role": "system", "content": verify_prompt},
            {"role": "user", "content": f"Claim: {query}\n\nSearch Results:\n{context}"}
        ])

        papers_dicts = [p.model_dump() for p in response.superseding_papers[:3]] 

        return {
            "claim_verdict": response.verdict_summary,
            "superseding_papers": papers_dicts,
        }
    except Exception as e:
        print(f"[verify_claim] ERROR: {type(e).__name__}: {e}")
        raise

async def generate_answer(state: RAGState):
    query = state["query"]
    route = state["route"]
    retrieved_docs = state.get("retrieved_docs") or []

    # 1. ROUTE: retrieve 
    if route == "retrieve":
        if not state.get("is_relevant"):
            reason = state.get("relevancy_reason", "No relevant information was found.")
            answer = (
                "I couldn't find information in the retrieved documents that "
                "answers your question.\n\n"
                f"Reason: {reason}\n\n"
                "You can try rephrasing the question or uploading additional documents."
            )

        else:
            # build context with citations from retrieved_docs
            context_parts = []
            for doc in retrieved_docs:
                source_type = doc.metadata.get("source_type")
                if source_type == "web":
                    title = doc.metadata.get("title", "Web Result")
                    url = doc.metadata.get("url", "")
                    context_parts.append(
                        f"[SOURCE]\nType: Web\nTitle: {title}\nURL: {url}\n\n"
                        f"Content:\n{doc.page_content}"
                    )
                else:
                    page = doc.metadata.get("page", 0) + 1
                    source = doc.metadata.get("title") or doc.metadata.get("source") or "Unknown Source"
                    context_parts.append(
                        f"[SOURCE]\nType: Document\nTitle: {source}\nPage: {page}\n\n"
                        f"Content:\n{doc.page_content}"
                    )
            context = "\n\n---\n\n".join(context_parts)
            prompt = f"""You are a research assistant.

            Answer the user's question STRICTLY and ONLY using the provided context.

            STRICT RULES:
            - Use ONLY information explicitly stated in the context.
            - Do NOT use outside knowledge.
            - Do NOT infer beyond the provided context.
            - For document sources use: [Source: <Title>, Page X]
            - For web sources use: [Source: <Title>]
            - Every factual paragraph must contain at least one citation.
            - Do NOT cite sources that are not present in the context.
            - If the context does not explicitly confirm or mention the specific thing asked about, 
              say clearly that it is not found in the document/context — do not answer based on 
              related or similar-sounding content.

            IMPORTANT — when multiple sources give slightly different values for the 
            SAME fact (e.g. temperature, price, score): give ONE clear primary answer 
            using the most authoritative/official source (e.g. government sources like 
            IMD over generic weather sites), then briefly note that other sources report 
            similar but slightly different values, without listing every single number.

            Context:
            {context}

            Question:
            {query}

            Answer:"""

            response = await get_kimi_llm(state).ainvoke([{"role": "user", "content": prompt}])
            answer = response.content
            # chunks = []
            # async for chunk in kimi_llm.astream([{"role": "user", "content": prompt}]):
            #     chunks.append(chunk.content)
            # answer = "".join(chunks)
            

    # 2. ROUTE: verify_claim 
    elif route == "verify_claim":
        verdict = state.get("claim_verdict", "")
        papers = state.get("superseding_papers") or []

        if papers:
            papers_block = "\n\n".join(
                f"{i+1}. **{p['title']}**\n   {p['summary']}\n   Link: {p['url']}"
                for i, p in enumerate(papers)
            )
            answer = (
                f"**Claim Verification Result**\n\n> {query}\n\n"
                f"**Verdict:** {verdict}\n\n**Superseding Papers:**\n\n{papers_block}"
            )
        else:
            answer = f"**Claim Verification Result**\n\n> {query}\n\n**Verdict:** {verdict}" 

    # ── ROUTE: direct_answer ────────────────────────────────────────
    else:
        response =await get_kimi_llm(state).ainvoke([
            {"role": "system", "content": "Answer using your general knowledge. Be concise."},
            {"role": "user", "content": query}
        ])
        answer = response.content
        # chunks = []
        # async for chunk in kimi_llm.astream([
        #     {"role": "system", "content": "Answer using your general knowledge. Be concise."},
        #     {"role": "user", "content": query}
        # ]):
        #     chunks.append(chunk.content)
        # answer = "".join(chunks)

    return {
        "answer": answer,
        "messages": [AIMessage(content=answer)]
    }

graph = StateGraph(RAGState)

graph.add_node("router",router)
graph.add_node("verify_claim",verify_claim)
graph.add_node("generate_answer",generate_answer)
graph.add_node("agent_node",agent_node)
graph.add_node("tool_node",tool_node)
graph.add_node("relevancy_check",relevancy_check)
graph.add_node("rewrite_query",rewrite_query)
graph.add_edge(START,"router")
graph.add_conditional_edges("router",route_decision,{"agent_node":"agent_node","verify_claim":"verify_claim","generate_answer":"generate_answer"})
graph.add_conditional_edges("agent_node",tool_relevancy_decision,{"tool_node":"tool_node","relevancy_check":"relevancy_check","generate_answer": "generate_answer"})
graph.add_edge("verify_claim","generate_answer")
graph.add_edge("tool_node", "agent_node")
graph.add_conditional_edges("relevancy_check",check_relevancy,{"generate_answer":"generate_answer","rewrite_query":"rewrite_query"})
graph.add_edge("rewrite_query","agent_node")
graph.add_edge("generate_answer",END)

# initial_state = {
#     "session_id": "123",
#     "messages": [HumanMessage(content="Is the claim that transformers outperform RNNs on all sequence tasks still valid?")]
# }


