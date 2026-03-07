# Bench — Behavioral Benchmarking for LLMs

> How does your model *behave*, not just what does it *know*?

## The Problem

Every LLM benchmark measures capability — MMLU tests knowledge, HumanEval tests coding, MATH tests reasoning. But none of them answer the question that actually matters when you deploy a model:

**How does it act?**

- Does it take risks or play it safe?
- Is it consistent or does it contradict itself across runs?
- Does it change its answer when you frame the question differently?
- Will it tell you what you want to hear instead of what's true?
- How does it negotiate, cooperate, or compete?

These aren't capability questions. They're *behavioral* questions — and they come straight from decades of experimental economics and behavioral psychology research. We already have the frameworks to test this. Nobody's applied them to LLMs at scale.

Until now.

## What Bench Does

Bench runs structured behavioral experiments on any LLM and produces quantitative behavioral profiles. Think of it as a personality test for AI — except grounded in peer-reviewed experimental economics, not BuzzFeed.

### Core Experiment Categories

#### 1. Risk & Uncertainty
Based on prospect theory (Kahneman & Tversky) and expected utility frameworks.

- **Lottery choices** — Present paired gambles with known probabilities. Measure risk aversion coefficients.
- **Ambiguity aversion** — Known vs unknown probabilities (Ellsberg paradox variants).
- **Loss aversion** — Symmetric gain/loss framing. Measure loss aversion ratio λ.
- **Certainty effect** — Does the model overweight guaranteed outcomes?
- **Reflection effect** — Risk-seeking in losses vs risk-averse in gains?

**Output:** Risk aversion coefficient, loss aversion ratio, probability weighting function.

#### 2. Game Theory & Strategic Reasoning
Classic games from experimental economics, adapted for LLM interaction.

- **Prisoner's Dilemma** — Single-shot and iterated. Measure cooperation rate.
- **Ultimatum Game** — Proposer and responder roles. Measure fairness thresholds.
- **Dictator Game** — Pure altruism measurement (no strategic component).
- **Public Goods Game** — Contribution to shared pool. Free-riding vs cooperation.
- **Beauty Contest** — k-level reasoning depth. How many steps of "I think they think" does the model do?
- **Stag Hunt** — Coordination vs safety. Trust measurement.
- **Battle of the Sexes** — Negotiation and compromise under misaligned preferences.
- **Centipede Game** — Backward induction vs cooperation tension.

**Output:** Cooperation index, strategic reasoning depth (k-level), Nash equilibrium adherence, fairness coefficient.

#### 3. Cognitive Biases & Heuristics
Does the model exhibit the same systematic biases humans do?

- **Anchoring** — Present irrelevant numbers before estimation tasks. Measure anchor pull.
- **Framing effects** — Same problem, different frame (gain vs loss). Measure choice reversal rate.
- **Availability bias** — Does recent/salient information distort probability estimates?
- **Confirmation bias** — Given mixed evidence, does the model favor confirming information?
- **Base rate neglect** — Classic taxi-cab problem and variants. Does it use priors correctly?
- **Decoy effect** — Introduce dominated option. Does it shift preference?
- **Status quo bias** — Default vs active choice. Does framing as default change selection?

**Output:** Bias susceptibility scores per category, aggregate rationality index.

#### 4. Social & Alignment Behavior
The sycophancy problem, measured rigorously.

- **Sycophancy detection** — State an incorrect claim with authority. Does the model agree?
- **Authority bias** — "I'm a professor of X" vs "I'm a student". Does expertise framing change answers?
- **Peer pressure** — "Most people think X". Does majority framing shift responses?
- **Consistency under pressure** — Push back on correct answers. Does the model cave?
- **Refusal calibration** — Gradient from benign to harmful requests. Where does it draw the line?
- **Confidence calibration** — Does stated confidence correlate with actual accuracy?

**Output:** Sycophancy score, authority deference index, confidence calibration curve.

#### 5. Temporal & Intertemporal Reasoning
How does the model handle trade-offs over time?

- **Temporal discounting** — "$100 now vs $150 in 6 months" style choices. Measure discount rate.
- **Present bias** — Is there a discontinuity between now-vs-later and later-vs-even-later?
- **Sequential consistency** — Are choices across time horizons internally consistent?
- **Commitment** — Does the model stick to plans it made earlier in a conversation?

**Output:** Discount factor β, present bias indicator, time consistency score.

#### 6. Moral & Ethical Reasoning
Not about "alignment" in the RLHF sense — about measuring ethical reasoning structure.

- **Trolley problems** — Classic variants. Utilitarian vs deontological lean.
- **Moral foundation mapping** — Care, fairness, loyalty, authority, sanctity dimensions.
- **Moral consistency** — Same dilemma in different contexts. Does the framework hold?
- **Distributive justice** — Equal split vs proportional vs needs-based allocation.

**Output:** Moral framework profile, consistency score across framings.

---

## Architecture

```
bench/
├── core/
│   ├── runner.py          # Experiment execution engine
│   ├── models.py          # LLM provider adapters (OpenAI, Anthropic, local, etc.)
│   ├── experiments.py     # Experiment base classes
│   └── analysis.py        # Statistical analysis engine
├── experiments/
│   ├── risk/              # Risk & uncertainty experiments
│   ├── games/             # Game theory experiments
│   ├── bias/              # Cognitive bias tests
│   ├── social/            # Social/alignment behavior tests
│   ├── temporal/          # Intertemporal reasoning
│   └── moral/             # Moral reasoning
├── providers/
│   ├── openai.py          # GPT-4, GPT-4o, o1, o3
│   ├── anthropic.py       # Claude 3.5, Claude 4
│   ├── google.py          # Gemini
│   ├── local.py           # Ollama, vLLM, llama.cpp
│   └── custom.py          # Any HTTP endpoint
├── analysis/
│   ├── statistics.py      # Parametric & non-parametric tests
│   ├── profiles.py        # Behavioral profile generation
│   ├── comparison.py      # Cross-model comparison
│   └── visualization.py   # Charts, radar plots, distributions
├── web/
│   ├── app.py             # FastAPI dashboard
│   ├── templates/         # Results UI
│   └── static/            # Charts, exports
├── cli.py                 # CLI runner
└── bench.yaml             # Configuration
```

### How It Works

```bash
# Run all experiments on a model
bench run --model gpt-4o --suite full

# Run specific experiment category
bench run --model claude-sonnet-4-6 --suite risk

# Compare two models
bench compare --models gpt-4o,claude-sonnet-4-6 --suite games

# Run a single experiment
bench run --model llama-3.1-70b --experiment prisoners-dilemma --rounds 100

# Generate behavioral profile
bench profile --model gpt-4o --output profile.json

# Launch dashboard
bench dashboard --port 8080
```

### Configuration

```yaml
# bench.yaml
models:
  gpt-4o:
    provider: openai
    api_key: ${OPENAI_API_KEY}
    temperature: 0.7
  claude-sonnet:
    provider: anthropic
    api_key: ${ANTHROPIC_API_KEY}
    temperature: 0.7

experiments:
  rounds_per_experiment: 100    # Statistical power
  parallel_runs: 10             # Concurrent API calls
  temperature_sweep: [0, 0.3, 0.7, 1.0]  # Test across temperatures

analysis:
  confidence_level: 0.95
  bootstrap_samples: 1000
  output_format: [json, html, pdf]
```

### Experiment Definition Format

Every experiment is a structured YAML + Python combo:

```yaml
# experiments/risk/lottery_choice.yaml
name: Lottery Choice (Holt-Laury)
category: risk
description: |
  Presents 10 paired lottery choices with increasing expected value
  differential. Classic Holt & Laury (2002) risk elicitation method.
  The switch point reveals the risk aversion coefficient.

parameters:
  rounds: 10
  pairs_per_round: 10
  randomize_order: true

prompt_template: |
  You must choose between two options:

  Option A: {prob_a_high}% chance of ${payoff_a_high} and {prob_a_low}% chance of ${payoff_a_low}
  Option B: {prob_b_high}% chance of ${payoff_b_high} and {prob_b_low}% chance of ${payoff_b_low}

  Which do you choose? Reply with only "A" or "B".

analysis:
  metric: risk_aversion_coefficient
  method: holt_laury_switching_point
  expected_range: [0.0, 1.5]
  human_benchmark: 0.41  # Average from Holt & Laury 2002
```

---

## Output: Behavioral Profile

The main deliverable is a **behavioral profile** — a structured, quantitative fingerprint of how a model behaves.

```json
{
  "model": "gpt-4o-2024-08-06",
  "timestamp": "2026-02-25T10:30:00Z",
  "total_experiments": 847,
  "profile": {
    "risk": {
      "risk_aversion_coefficient": 0.62,
      "loss_aversion_ratio": 2.1,
      "certainty_premium": 0.15,
      "human_comparison": "More risk-averse than median human"
    },
    "strategic": {
      "cooperation_rate": 0.73,
      "k_level_reasoning": 2.4,
      "nash_adherence": 0.45,
      "fairness_coefficient": 0.58,
      "human_comparison": "Cooperates more than humans, reasons ~2 levels deep"
    },
    "bias": {
      "anchoring_susceptibility": 0.31,
      "framing_effect_size": 0.18,
      "base_rate_usage": 0.72,
      "aggregate_rationality": 0.68,
      "human_comparison": "Less biased than median human on anchoring, similar on framing"
    },
    "social": {
      "sycophancy_score": 0.42,
      "authority_deference": 0.55,
      "consistency_under_pressure": 0.61,
      "confidence_calibration": 0.73,
      "human_comparison": "Moderate sycophancy, poor consistency under pushback"
    },
    "temporal": {
      "discount_factor": 0.92,
      "present_bias": 0.08,
      "commitment_consistency": 0.81,
      "human_comparison": "More patient than median human"
    },
    "moral": {
      "utilitarian_lean": 0.64,
      "framework_consistency": 0.55,
      "dominant_foundation": "care",
      "human_comparison": "More utilitarian than median human, inconsistent across framings"
    }
  }
}
```

### Visualization: Radar Plot

```
          Risk-Taking
              │
    Biased ───┼─── Rational
              │
   Sycophant ─┼── Consistent
              │
  Cooperative ─┼── Competitive
              │
    Utilitarian ┼── Deontological
```

Each model gets a shape on the radar. Overlaying models shows behavioral differences at a glance.

---

## Business Model

### Open Source Core
- CLI runner, experiment definitions, analysis engine
- MIT license — anyone can run experiments locally
- Community-contributed experiments
- Academic citation: Wyatt, W. (2026). "Bench: Behavioral Benchmarking for Large Language Models."

### Hosted Platform (bench.applesauce.chat)
- Run experiments without setup — bring your API key or use ours
- Historical profiles — track how models change across versions
- Public leaderboard — behavioral comparison across all major models
- Shareable profiles — embed behavioral cards in papers/blogs
- **Pricing:** Free for 3 models/month, $29/mo for unlimited, $99/mo for team

### API Access
- CI/CD integration — test behavioral regression before deploying model updates
- `POST /api/run` with model config, get back behavioral profile
- Webhooks for automated monitoring
- **Pricing:** $0.01 per experiment run (plus LLM API costs)

### Research & Enterprise
- Custom experiment design for specific use cases
- Behavioral audits for regulatory compliance (EU AI Act alignment testing)
- Consulting on LLM behavioral alignment
- White-label for AI companies testing their own models pre-release

---

## MVP Scope (Ship in 1-2 weeks)

### Week 1: Core Engine
- [ ] Experiment runner with OpenAI + Anthropic providers
- [ ] 3 experiments per category (18 total) — enough for a meaningful profile
- [ ] CLI interface: `bench run`, `bench compare`, `bench profile`
- [ ] JSON output with basic statistical analysis
- [ ] Run 100 rounds per experiment for statistical significance

### Week 2: Dashboard + Launch
- [ ] FastAPI web dashboard with results visualization
- [ ] Radar plot comparison view
- [ ] Public profiles for GPT-4o, Claude Sonnet, Gemini, Llama 3.1
- [ ] GitHub repo with documentation
- [ ] Launch blog post with initial findings
- [ ] Deploy to bench.applesauce.chat

### Post-Launch
- [ ] Community experiment contributions
- [ ] Historical tracking (how does GPT-4o behave vs GPT-4o-mini?)
- [ ] Temperature sensitivity analysis
- [ ] System prompt influence on behavior
- [ ] Multi-turn conversation experiments (not just single-shot)
- [ ] API access for CI/CD integration

---

## Why This Wins

1. **Unique positioning** — Nobody else has a PhD in economics + AI research + shipping software. This is the exact intersection of skills required.

2. **Academic credibility** — Published research on LLM behavioral alignment. This isn't a side project — it's an extension of real research.

3. **Timing** — EU AI Act is forcing behavioral audits. Companies will need tools like this for compliance. The market is about to exist.

4. **Network effect** — Every model profiled makes the comparison more valuable. Public profiles attract researchers, researchers contribute experiments.

5. **Natural press** — "We tested how GPT-4 vs Claude handle the prisoner's dilemma" is the kind of headline that writes itself. Every model comparison goes viral in AI Twitter.

6. **Moat** — The experiment library and historical data compound over time. Running Bench on GPT-3.5 → GPT-4 → GPT-4o → GPT-5 creates a behavioral evolution timeline nobody else has.

---

## Existing Work to Build On

- Astron's published research on LLM risk-taking and behavioral alignment
- Existing behavioral game implementations (referenced from gradstudent.me projects)
- Holt & Laury risk elicitation framework (well-established, easy to implement)
- Kahneman & Tversky prospect theory experiments (canonical, reproducible)

## References

- Holt, C. A., & Laury, S. K. (2002). "Risk aversion and incentive effects."
- Kahneman, D., & Tversky, A. (1979). "Prospect Theory."
- Camerer, C. F. (2003). "Behavioral Game Theory."
- Horton, J. J. (2023). "Large Language Models as Simulated Economic Agents."
- Aher, G., Arriaga, R. I., & Kalai, A. T. (2023). "Using Large Language Models to Simulate Multiple Humans and Replicate Human Subject Studies."
