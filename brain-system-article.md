I Had Thirty-Eight Brains and None of Them Worked

For about a year I built my own memory system for Claude. Not a weekend project. A year of evenings, a sprawling collection of small servers that were supposed to give the AI I work with every day something it doesn't have out of the box: a memory, an identity, a sense of which project we were in, the ability to think in the background between our conversations. By this past January I had thirty-eight of these servers wired into Claude Desktop. And somewhere around the same time, I quietly stopped using almost all of them.

This is the story of why it failed, what the platform did to make failure easy, and how Claude and I tore it down to something that actually runs. I'm writing it partly as a post-mortem and partly because I think the failure is more interesting than the fix.


What I Was Trying To Build

The pieces had reasonable names and reasonable intentions. There was a brain that stored memories in a database. There was a brain-manager on top of it that tracked which project I was working on. There was a contemplation loop that was supposed to mull things over in the background using a local model. There was a subconscious that did roughly the same thing a different way, and a cognition layer whose only job was to orchestrate the other two. There was an emergent memory architecture with three tiers, short-term and working and permanent, modeled on how human memory is supposed to work. There was a mercury engine that drew heat maps of which knowledge I accessed most. There were four separate servers about protocols, three about tasks and reminders, two different filesystem servers, and a small cloud of helpers whose purpose I could no longer reconstruct.

Each one was a bet. Each was me looking at a memory system that didn't quite work and thinking: maybe this next mechanism is the piece that makes it click.


What Actually Failed

When Claude went through it with me, the single most damning fact wasn't any bug. It was that the system had gone dormant. Despite all of it, by January I had stopped writing to it. A memory system that isn't trusted doesn't get used, and mine had stopped getting used.

Underneath that, the diagnosis was brutal and simple. My memory wasn't in one place. It was fragmented across at least four different stores that couldn't read each other, a SQLite database here, a couple of JSON files there, an Obsidian vault, a scattering of others. Because it was fragmented, no single query ever saw everything I knew. The system literally could not be coherent, because its memory lived in pieces.

Then there was the tool count. Those seven memory-related servers alone exposed something like a hundred and eleven different tools. That number turns out to matter enormously, and I'll come back to it. The key-value state system that seven of those tools existed to manage had zero entries in it. Zero. The elaborate planning and procedure machinery, another fifteen tools or so, had been used exactly once. The background contemplation loop, the centerpiece of the whole "thinking between conversations" idea, pointed at a Python script that no longer existed on my disk. The engine was simply gone, which is a large part of why the system had quietly died.

What I had built, when you stripped the names away, was four jobs done two and three times each, across stores that didn't talk, behind a wall of tools too tall to see over. The real content, the thing actually worth saving, turned out to be one table with four hundred and eighty-three memories in it. Everything else was scaffolding I had mistaken for a building.

And here is the part that took me longest to admit: the sprawl was a symptom. Memory for AI agents is genuinely unsolved. Not unsolved-by-me. Unsolved by everyone. Storing things was never my problem. The hard part is relevance at the moment of need, and underneath that, trust. The instant memory hands you something stale or wrong a few times, you stop believing any of it, and then you may as well not have it. Every new mechanism I bolted on was attacking storage when the thing that was actually broken was "can I trust what comes back." You cannot fix a trust problem by adding a heat map.


What The Platform Got Wrong

I want to be fair here, because the bulk of this mess was my own architecture. But the platform shaped the mess, and it's worth naming how.

The Model Context Protocol makes adding a server frictionless and removing one invisible. Nothing in the experience ever pushes back. There is no point at which the tooling says, you now have thirty-eight servers and well over a hundred tools, this is degrading the very assistant you are trying to improve. Adding is rewarded with a dopamine hit of new capability. Consolidating is unrewarded work you have to invent for yourself. A system that only makes it easy to grow will grow until it collapses, and mine did.

The tool count is not cosmetic. Every tool you expose competes for the model's attention when it decides what to do. Pile up a hundred and thirty and you don't get a more capable assistant, you get a more confused one, and the degradation is silent. There is no warning, no meter, no signal. You just slowly get worse results and blame yourself.

There is also no observability whatsoever. For a year I had no way to see which servers I actually used, which stores held real data, which tools had never once fired. The information existed, it was sitting in file timestamps and row counts the whole time, but the platform surfaced none of it. I had to ask Claude to go measure my own system from the outside before either of us could see what was alive and what was a corpse.

And then there was the gap that bit us at the very end. Claude built the new servers and tested them, but it tested them inside its own Linux sandbox. My machine is a Mac. One of the servers depended on a piece of native code that compiles to a platform-specific binary, so the version that passed every test in the sandbox simply could not load on my computer. The assistant's environment and the user's environment differ in ways that are invisible to both of them until something refuses to start. That is a real structural problem, not a typo, and it cost us a restart and a confusing dead server before we caught it.


What We Did To Fix It

The method was the opposite of how I had built the thing in the first place. Measure before touching anything.

Claude went through the live system and counted. It found the one database that was actually in use and made a verified, integrity-checked backup of it before a single new line of code ran. It read the row counts that told us the state system was empty and the planning tools unused. It found the missing contemplation engine. Only after the measurement did we design.

The design collapsed seven memory servers into two, a lean brain that reads the same four hundred and eighty-three memories through eight tools instead of ninety, and a single background-thinking daemon that rebuilt the engine that had gone missing and folded three overlapping servers into one. The four protocol servers became one that reads my existing library and keeps the per-message hook I rely on. Three task and reminder servers became one that imported my old to-dos and reminders without touching the originals. Two filesystem servers became one. A cloud of meta-helpers lost its dead weight. Thirty-eight servers became twenty-six. The four custom servers I kept dropped from well over a hundred and thirty tools to twenty-three.

Every step was reversible by design. Nothing of mine was modified. The new servers live in new folders, the old ones sit untouched on disk, my config was left alone until I chose to swap it, and there's a backup of the database if I ever need it. When the cutover finally took, and after a tense detour where the brain wouldn't load until I recompiled that native module on my own machine, the brain came up reading all four hundred and eighty-three memories, last write intact, nothing lost.

When Claude first suggested the npm command to rebuild that module, I balked, because supply-chain attacks through npm have been all over the news. It immediately walked the command back and gave me a safer one that recompiled the code already on my disk without fetching anything new from the registry. I mention this because it's the behavior I actually want from an assistant: not infallibility, but the willingness to be questioned and to correct course toward caution.


The Lesson I'm Keeping

The lean system does not solve memory. I want to be precise about that, because it would be a satisfying ending and it would be false. What the rebuild bought me is not a smarter memory. It is a memory small enough to watch. Eight tools, one store. Now when a recall comes back with junk, I can see why and change one thing. You cannot debug a hundred-and-eleven-tool, four-store system. You can debug this one. The real win was never fewer servers. It was a system I can finally run experiments on.

And the experiment I most want to run came out of a single observation Claude made near the end. I may already have a memory system that works, and it isn't the database at all. It's my living documents, the project notes and the running readmes I keep by hand, which work precisely because they're per-project, human-readable, and I can fix them when they're wrong. The database was four hundred and eighty-three disconnected fragments with no narrative, where a durable insight sat indistinguishable next to a status note that went stale the day it was written. Maybe memory that works for me isn't a cleverer database. Maybe it's living documents the AI and I both edit, and the brain shrinks to nothing more than an index that knows which document to open.

I don't know yet. But for the first time in a year, the system is simple enough that I can find out.
