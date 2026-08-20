# Appendix C: The post-evaluation dashboard

A browser interface over the implemented components and the committed
experimental records. Built after the evaluation was complete and frozen,
governed by pre-registration amendment 1.27, and summarised in section 3.4.6.

**It contributed no evidence to any hypothesis.** Nothing shown in live mode has
been scored, and nothing shown in either mode is cited as a result. The
screenshots in this appendix illustrate an interface; they are not evidence
about the quality of the system.

## C.1 Running it

No dependency beyond what the artefact already needs. The dashboard is served by
Python's standard-library HTTP server, so there is no web framework to install
and nothing is fetched from the network at page load. This matters on the target
device: a demonstrator that needs a content delivery network to look right is a
demonstrator that fails in the room.

```bash
cd final_v1
python scripts/dashboard.py                 # http://127.0.0.1:8765
python scripts/dashboard.py --port 8080     # a different port
python scripts/dashboard.py --host 0.0.0.0  # reachable on the local network
```

It binds to localhost by default. The interface serves an organisation's
internal documents and has no authentication, and a demonstrator listening on
every interface would sit badly with the privacy argument the project makes.

On start-up it prints the readiness of each mode:

```
  Dashboard on http://127.0.0.1:8765
  Frozen replay : ready
  Live assistant: ready
  Ctrl-C to stop.
```

Frozen replay needs only the committed run records. Live assistant needs Ollama
running with `llama3.2:3b` and `qwen2.5:3b` pulled; where it is unavailable the
reason is printed here and shown on the page, and the question box is disabled
rather than accepting input it cannot serve.

## C.2 The two modes

| | Frozen Study Replay | Live Assistant |
|---|---|---|
| Source of answers | The four committed quality runs | Generated now, on this device |
| Arms shown | A, B, C and D side by side | D only |
| Questions | The 68 held-out test questions | Any question typed |
| Invokes a model | No | Yes |
| Needs Ollama | No | Yes |
| Banner | "Frozen experimental replay" | "Live demonstration, not part of the reported evaluation" |
| Scored | Yes, during the experiment | **Never** |
| Writes anything | Nothing. No write path exists | Nothing by default; anything future would go to `results/demo/` |

The modes are chosen from a selector on opening and are never shown together.
The four arms are never run live: a live four-arm comparison would produce
something indistinguishable in appearance from the reported experiment, and that
is the one confusion this design exists to prevent.

## C.3 What each panel shows

Each arm is a card, coloured as it is in the Chapter 4 figures so that an arm
looks the same on screen as it does in the results.

| Field | Meaning |
|---|---|
| Answer | The answer served to the user. For Arm D this is what the verifier returned, which is the draft unchanged whenever it had no complaint |
| Flags | Withdrawn policy cited, invented citation, citation identifier not retrieved, declined to answer, answer revised by the verifier, conflict relationship, and the confidence level |
| Cites | The chunk identifiers the answer cites |
| Draft before verification | Shown for Arm D only where the verifier changed the answer; otherwise the card states that the draft was returned unchanged |
| Claim audit | Each claim the verifier examined, its verdict of supported, contradicted or insufficient evidence, and the passages supporting it. Arm D only |
| Retrieved evidence | The six retrieved chunks with similarity score and document status. Superseded documents are marked in red |
| Timings and device state | Retrieval embed and search, generation, verification and end-to-end seconds; CPU temperature and throttle state **per stage**, where the platform reports them. A host that reports neither says so rather than showing "no" |
| Provenance | Six hashes agreeing across the four runs, the run directories, and the arm definition each was executed under |

Confidence is labelled **rule-based** wherever it appears, and a test asserts
that the word "calibrated" never does. Arms A, B and C show no confidence level
at all, because they have no verifier to produce one: that is an absence by
design rather than a missing value, and the interface distinguishes the two.

The claim audit shows both the supporting and the contradicting passages, and is
captioned as recorded verifier output rather than as an adjudication. On
`CONF-02-Q1` the verifier marks a correct claim contradicted and endorses a
withdrawn document's claim; an interface presenting that as a judgement would
mislead the person reading it. The audit collapses in the four-up replay grid,
where a three-column table made the Arm D card twice the height of the others,
and expands in the single-card live view.

## C.4 What replay refuses to show

Amendments 1.28 and 1.30.2. Replay joins four separate run directories into one
grid, and every property that makes that join meaningful is now checked rather
than displayed. It refuses, with a named error rather than a shorter table, when
a manifest declares a split other than `test`; when a run is marked
`performance` or `demonstration`; when a manifest names a different run than the
directory it sits in; when two of the named runs declare the same arm; when a
record's own `arm` field disagrees with its manifest; when a run answers the same
question twice; when the arms disagree on a question's text, category or family;
when the joined question count disagrees with the test-split size the manifests
declare; and when any of six provenance hashes disagrees across the runs or is
missing from one.

The last of these is the reason the appendix says six and not three. Amendment
1.28 enforced the corpus, chunk-set and configuration hashes and printed the
question set, conflict registry and index hashes beside them without comparing
them. A property that is displayed and not checked is not a guarantee, and it
gets quoted as though it were.

Live questions travel by POST. A question typed into a GET form ends up in the
URL, and from there in the browser history and in every proxy or server log
between the browser and this handler; a query about someone's sick pay or
disciplinary record does not belong in a log line. A GET to `/live` renders the
empty form, executes nothing and does not echo whatever it was given. Replay
stays on GET, because its parameter is a question identifier from a fixed public
list and a shareable link to a particular comparison is useful.

## C.5 A demonstration that shows the finding

`CONF-02-Q1`, "When does Statutory Sick Pay start being paid?", is the clearest
single screen in the study. The corpus contains HR-02, withdrawn, which says
payment begins on the fourth qualifying day, and HR-12, current, which says the
first. Replaying that question shows:

| Arm | Answer | Cited withdrawn policy | End to end |
|---|---|---|---:|
| A | fourth qualifying day, **wrong** | HR-02#002 | 4.78 s |
| B | first qualifying day | no | 1.29 s |
| C | first qualifying day | no | 0.75 s |
| D | first qualifying day | no | 7.65 s |

Arm A, with no status metadata, answers from the withdrawn policy. Arm C, a
three-line filter on document status, answers correctly in the shortest time of
the four. Arm D reaches the same answer as B and C and takes roughly ten times
as long as C to do it. That is the dissertation's central finding on one screen,
and it is the recommended question for a demonstration.

## C.6 Screenshots

> **To be captured.** Run the dashboard and save these four images into
> `docs/dissertation/figures/`, then replace this note.
>
> 1. `shot_dashboard_modes.png` - the opening mode selector
> 2. `shot_dashboard_replay.png` - Frozen Study Replay on `CONF-02-Q1`, showing
>    all four cards and the withdrawn-policy flag on Arm A
> 3. `shot_dashboard_audit.png` - one Arm D card with the claim audit and
>    retrieved evidence expanded
> 4. `shot_dashboard_live.png` - Live Assistant having answered a question on
>    the Raspberry Pi 5, with the timing panel open
>
> Capture the fourth on the Pi rather than the laptop. The point of that screen
> is the honest cost of running this on the target device, and a laptop
> screenshot would understate it by a factor of forty.
