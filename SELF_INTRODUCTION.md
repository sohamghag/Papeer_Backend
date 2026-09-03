# Self-Introduction & Opening Round

**Soham Ghag** — final-year Metallurgical & Materials Engineering, NIT Warangal.
Interviewing for a **backend engineer** role.

Your background is a differentiator, not a liability. Lead with it, don't apologise for it.

---

## 1. THE FINAL INTRO — FISCHERJORDAN ⭐ USE THIS ONE

~55 seconds. Use for "Tell me about yourself."

> "Good evening, and thank you for this opportunity. My name is Soham Ghag, and I'm in my final year of Metallurgical and Materials Engineering at the National Institute of Technology, Warangal.
>
> Alongside my degree I've been freelancing as a mobile app developer for the last eight months. I built an Android app for merchant navy aspirants with a Node and Express backend, and it's live on the Play Store with over six thousand downloads.
>
> I've also worked on two main projects. The first is Papeer — a retrieval system that lets you load your own documents and query them in plain language, with answers grounded in the source material and cited back to it. It's built on FastAPI with hybrid search over a vector database, and it's deployed and running.
>
> The second came out of my own field — a machine learning pipeline predicting CO₂ adsorption in metal-organic frameworks, trained on a 137,000-structure dataset.
>
> I'm here for backend engineering specifically. And to be honest about why this company — I read about your Domain Expert work, and it is the same problem I have been working on for the last year.
>
> Outside academics, I play table tennis and kho kho."

### Delivery notes

- Say the FischerJordan line **evenly, not eagerly**. It's a strong claim — it works because it sounds like a plain statement of fact.
- It's a deliberate hook. Expect her to ask about Papeer, hybrid search, or what you thought of their Domain Expert work. All three are your strongest ground.
- If she asks **"where do these systems break?"** — that's the best minute you have:
  1. Retrieval misses exact terms → hybrid BM25 + dense, recall 0.77 → 0.86
  2. Compound questions return nothing → query expansion + rank fusion
  3. Ingestion destroys tables and scanned pages → atomic table chunks, OCR, reject-if-empty
  4. Underneath all three: *the system looks like it's working*. That's why mine says "I couldn't find this" with a reason instead of falling back on general knowledge.

### Why it's structured this way

| Beat | Job it does |
|---|---|
| NIT Warangal, stated clearly | Recognised name — don't mumble it |
| **Papeer first** | Signals "backend engineer" immediately; MOF then reads as range, not as your identity |
| Two hooks, low detail | Gives them a choice of what to pull on instead of a monologue |
| "Came out of my own field" | Makes the pivot deliberate, not random |
| Closing on why backend | You're choosing them, not spraying applications |
| Sports, **last**, one line | Human close. Not in the middle — it isn't part of your professional case |

**If they follow up on the sports:**

> "Kho kho especially — I've played it since school. It's a team sport that's almost entirely about coordination and reading the other side, so it's quite different from table tennis, which is individual and reaction-based. I like having both."

Good answer because it shows *why* you like each, rather than just naming them.

**Deliberately left out:** LangGraph, Qdrant, Postgres, LangSmith, the ingestion-cost bug. All of that lands harder as an *answer* than as part of the opening. Hold it.

---

## 2. THE LONGER VERSION (~55 seconds)

Only if they explicitly ask for more, or the room is clearly waiting.

> "...The first is Papeer — a full-stack RAG application with a FastAPI backend, where users upload documents and have a conversational chat with them. It uses LangGraph for orchestration, hybrid search over Qdrant, and Postgres for session persistence. It's deployed and running. Building it gave me hands-on experience with async Python, API design, vector databases, and observability — I benchmarked the whole pipeline with LangSmith tracing, which is how I caught a bug that halved my ingestion cost.
>
> The second came out of my own field — a machine learning pipeline predicting CO₂ adsorption in metal-organic frameworks, which are candidate materials for carbon capture. I trained it on a 137,000-structure dataset and found that domain-informed feature engineering improved accuracy about twenty times more than hyperparameter tuning did.
>
> Backend engineering is where I want to build my career, which is why I'm here."

---

## 3. THE VERY SHORT VERSION (~25 seconds)

If they say "briefly" or you sense time pressure:

> "Good morning, and thank you for this opportunity. I'm Soham Ghag, final year Metallurgical and Materials Engineering at NIT Warangal.
>
> I worked on two main projects. Papeer — a full-stack RAG application with a FastAPI backend, deployed. And a machine learning pipeline predicting CO₂ uptake in metal-organic frameworks, trained on 137,000 structures.
>
> Backend is where I want to build my career, which is why I'm here."

---

## 3. "WALK ME THROUGH YOUR RESUME"

Different question — chronological, not thematic. Don't just re-read the intro.

> "I'm in my final year of Metallurgical and Materials Engineering at NIT Warangal. About two years ago I got into machine learning through my own field — I built a regression pipeline predicting CO₂ uptake in metal-organic frameworks using a 137,000-structure dataset, comparing XGBoost against a neural network. The interesting result was that domain-informed feature engineering improved R² about twenty times more than hyperparameter tuning did.
>
> That project pulled me toward the engineering side, so I moved into backend. I built Papeer, which started as a Streamlit prototype and I rebuilt it properly with FastAPI and React — LangGraph for orchestration, Qdrant for hybrid vector search, Supabase Postgres for session persistence, deployed on Render and Vercel.
>
> Alongside that I've picked up Docker and AWS EC2 from deploying the earlier version, and I'm comfortable with async Python, API design, and observability tooling.
>
> Happy to go deeper on either project."

---

## 4. ⚠️ THE QUESTION YOU WILL GET

**"You're a materials engineer. Why backend? Why should we hire you over a CS grad?"**

Do not get defensive. Do not over-apologise.

> "Honestly, the CS grad has four years of formal fundamentals I don't. What I have instead is that everything I know, I learned because I needed it for something I was building — so it stuck differently.
>
> Concretely: I didn't learn about ASGI from a lecture, I learned it because my requests take up to 21 seconds waiting on external APIs and I needed to understand why a synchronous framework would fall over under that. I didn't learn about connection poolers abstractly — I learned because Postgres connections failed on Render and it turned out Supabase's direct endpoint is IPv6-only.
>
> I'd also say the domain background isn't nothing. On the ML project, the biggest performance gain came from feature engineering informed by materials chemistry, not from hyperparameter tuning — about twenty times more improvement. That only happened because I understood the physics of what I was modelling. Being able to understand a problem domain and talk to the people who own it is a real part of backend work."

**That last paragraph is the strongest thing you have** — a measured result proving domain knowledge produced better engineering.

### If they push: "but do you know DSA / OS / networks?"

> "Not to the depth of a CS curriculum, and I won't pretend otherwise. I've been filling gaps in the areas my work touches — HTTP, concurrency models, database connection handling, indexing. Data structures and algorithms I'm actively practising. If there's a specific area that matters for this role, I'd rather you tell me now so I can be honest about where I stand."

Offering to be tested is more convincing than claiming competence.

---

## 5. OTHER OPENING-ROUND QUESTIONS

### "Why backend specifically?"

> "Two reasons. I like problems where you can measure whether you were right — latency, cost, throughput are all numbers, and they regularly told me my intuitions were wrong. And I like that backend is where the actual system design happens: how data is stored, how failure is handled, what happens under load. On Papeer, the interesting work wasn't making it work, it was making it terminate correctly when the model misbehaved and fail honestly when retrieval found nothing."

### "What's your greatest strength?"

> "I instrument things before I optimise them. On Papeer I had LangSmith tracing on every step with token counting, and that caught a bug where every embedding batch was being computed twice — it halved my ingestion cost. I'd never have found it by reading code. I assumed embedding was the expensive part of ingestion; the traces showed parsing takes twice as long. Almost every optimisation I'd have guessed at would have been aimed at the wrong thing."

### "What's your weakness?"

Pick a real one with a real correction. Never "I work too hard."

> "Formal CS fundamentals — data structures, algorithms, operating systems. I learned backend by building, which means I know the parts I've hit and have gaps in the parts I haven't. I'm working through DSA practice specifically because I know it's the area where a project-based path leaves holes."

**Or:**

> "I tend to keep building rather than stopping to validate. On the ML project I had a clean evaluation pipeline in v1, and when I rebuilt the RAG app I kept shipping features without rebuilding the eval harness — so right now I can't prove that a retrieval change actually improved quality, I'm going on intuition. I know that's the wrong order and it's the first thing on my list."

Both are honest, specific, and paired with an action.

### "Where do you see yourself in 5 years?"

> "Still building systems, but with a lot more depth on the infrastructure side — distributed systems, scaling, the things I've only touched at toy scale. I'd like to be the person on a team who's trusted with the hard architectural calls, and that takes years of seeing what breaks in production, which is exactly what I don't have yet."

### "Why this company?"

**Research them tonight. Non-negotiable.** Have one specific thing.

Template:

> "Two things. [Specific product / technical problem they work on] is genuinely the kind of problem I want to work on — [one sentence connecting it to something you've built]. And practically, I'm at the stage where I need to learn from people more experienced than me, and [something about their team, engineering blog, scale, or culture] suggests that's available here."

**Never** say "good learning opportunity" and stop. That's what everyone says.

### "Tell me about a challenge you faced."

> "On Papeer, retrieval quality. Getting a RAG demo working is easy; getting it to reliably find the right chunk is not. I went through three iterations. Dense-only vector search first — then I benchmarked it with DeepEval and found it was missing exact keyword matches, so I added BM25 sparse search and fused the results, which improved contextual recall by about twelve percent relative. Then I found compound questions were returning nothing at all, because a single embedding of a long multi-part question gets too diluted to match anything well. So I added query expansion — generate variations, retrieve for each, fuse the rankings. A test question that returned zero documents before returned a fully cited answer after.
>
> The lesson was that each fix only came from measuring the failure mode, not from guessing at improvements."

### "Tell me about a time you failed."

> "I tried to add token-by-token streaming to Papeer and had to revert it. LangGraph's streaming interacts awkwardly with the checkpointer and my node structure, and after a couple of days I had something half-working that broke conversation persistence. I made the call to ship the working non-streaming version rather than a broken streaming one. It's still on my roadmap, but I'd approach it by understanding the checkpointer's streaming semantics first rather than trying to bolt it on."

---

## 6. QUESTIONS TO ASK THEM

Always have three. Not asking any reads as disinterest.

**Good ones:**

- "What does the first six months look like for someone in this role?"
- "What's the biggest technical challenge the team is dealing with right now?"
- "How does the team handle code review and deployment — what's the path from merge to production?"
- "What's something that surprised you about working here?"
- "For someone coming from a non-CS background, where would you expect the steepest learning curve to be?"

**Avoid:** salary and leave policy in a technical round. Save those for HR.

---

## 7. RULES

**Don't** list technologies. "I know Python, FastAPI, Docker, AWS..." is a resume read-back — they already have it.

**Don't** apologise for the background. Never say "I'm from chemical engineering so I'm still learning." State it flatly and move on.

**Don't** go past 90 seconds. If they want more, they'll ask.

**Do** end on why *this* role. It signals you're choosing them, not spraying applications.

**Do** name one measurable thing. "Halved my ingestion cost" is worth more than three sentences of adjectives.

**Do** practise it out loud three times. Not memorised word for word — you want the beats, not a script. Reciting sounds worse than improvising.

---

## 8. THE 30-SECOND PRE-INTERVIEW CHECK

Before you walk in, be able to say these without thinking:

1. **Who you are** — final-year chemical engineering, backend focus, two years of building
2. **Your headline project** — Papeer: FastAPI + LangGraph RAG, deployed
3. **Your one number** — halved ingestion cost via LangSmith tracing; or R² +0.093 from feature engineering vs +0.004 from tuning
4. **Your pivot story** — MOF project pulled you from chemistry into engineering
5. **Why backend** — you like problems where measurement tells you you were wrong
