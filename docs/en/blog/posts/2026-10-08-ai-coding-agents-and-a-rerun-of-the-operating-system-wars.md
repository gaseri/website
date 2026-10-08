---
author: Vedran Miletić
authors:
  - vedranmiletic
date: 2026-10-08
tags:
  - artificial intelligence
  - decentralization
  - free and open-source software
  - information revolution
  - linux
  - freebsd
  - microsoft
---

# AI coding agents and a rerun of the operating-system wars

---

![brown train vehicle machine](https://unsplash.com/photos/_HRi5kBwGh0/download?w=1920)

Photo source: [Rod Long (@rodlong) | Unsplash](https://unsplash.com/photos/brown-train-vehicle-machine-_HRi5kBwGh0)

---

Attending the [Agentic AI for Science workshop](https://www.mpcdf.mpg.de/events/46569/14192) at MPCDF in Garching got me thinking about the platforms we are starting to build our scientific workflows around. The [workshop programme](https://plan.events.mpg.de/e/ai-agents-for-science) brings together hands-on work with agents, examples from scientific software development and materials simulations, and perspectives from industry. For someone interested in free and open-source software, this raises some familiar questions: who controls those platforms, what can we change ourselves, and how much of our work can we take elsewhere?

What interests me here is the software around the model. An *agent harness* gives the model tools, supplies context, and controls what it can do. The flagship GPT, Claude, and Gemini models are proprietary, but the software using them does not have to be (and often isn't). We can use the same model through different harnesses, with different licenses, tools, and restrictions. Choosing the model, deciding where it runs, and choosing the harness are separate decisions, even when a product offering bundles them together.

A few harnesses and development environments for programming, scientific workflows, and other tasks illustrate these differences. This is only part of the picture: agents embedded in enterprise applications bring their own integrations and dependencies.

This led me to a somewhat loose comparison: [Claude Code](https://code.claude.com/docs/en/overview) reminds me of Windows, [Codex CLI](https://learn.chatgpt.com/docs/codex/cli) of BSD, [OpenCode](https://opencode.ai/) and the like of Linux, and [Antigravity](https://antigravity.google/) of the Macintosh. Each comparison emphasizes something different: compatibility, permissive reuse, independence from suppliers, or the appeal of an integrated experience.

If you have used Claude Code in a terminal, the first part probably sounds wrong already. How does a program that reads files, invokes shell tools, and fits into a Unix workflow resemble Windows? Bear with me. The resemblance is in how people build tools and workflows around the software, rather than how its interface looks.

<!-- more -->

## Claude Code and the Windows platform

Windows was proprietary, but Microsoft very much wanted other developers to write software for it. The more applications were available, the more useful Windows became, and the more reasons users had to keep using it. Developers, in turn, had a reason to support Windows because that was where their users were. Having everyone build around your platform is a powerful position to be in.

I see a similar possibility with Claude Code. Anthropic controls the product, which is distributed under [its commercial terms](https://github.com/anthropics/claude-code/blob/main/LICENSE.md), while other developers can extend it through [plugins containing skills, agents, hooks, and MCP servers](https://code.claude.com/docs/en/plugins). Those servers expose tools and data to agents through the shared [Model Context Protocol (MCP)](https://modelcontextprotocol.io/docs/getting-started/intro). A proprietary product can attract a great deal of third-party software. Windows certainly did.

The more interesting part is what happens outside Claude Code itself. Suppose you maintain a competing harness and a user points it at a repository containing a `CLAUDE.md` file. You could ask the user to copy the instructions into your preferred format, but reading the existing file is more convenient. Sure enough, [OpenCode supports `CLAUDE.md` as a fallback when no `AGENTS.md` exists](https://opencode.ai/docs/rules/#claude-code-compatibility), along with other Claude Code conventions.

Once enough people write instructions and extensions for one product, supporting those conventions becomes useful even to its competitors. **The compatibility requirement can outlive the decision to use the original product.** That is the Windows part of the comparison. The terminal interface is beside the point.

The advantage of existing tool support also influenced me when [choosing between Markdown and reStructuredText for teaching materials](2021-08-16-markdown-vs-restructuredtext-for-teaching-materials.md). I preferred reStructuredText, but Markdown was supported by the tools my students and colleagues already used. The technically preferable option, from my perspective, lost to the one that was easier for everyone to work with.

## Codex, BSD, and permissive licensing

[Codex](https://openai.com/codex/) requires a little more care, since the name covers several different things. Here I mean [Codex CLI](https://learn.chatgpt.com/docs/codex/cli), whose code is available under the [Apache 2.0 license](https://github.com/openai/codex/blob/main/LICENSE). [OpenAI's overview](https://learn.chatgpt.com/docs/open-source) distinguishes the open-source components from the proprietary ones.

Furthermore, Codex CLI can use [custom providers and local models through Ollama or LM Studio](https://learn.chatgpt.com/docs/config-file/config-advanced). The endpoint and model need to be compatible with what the harness expects, but the CLI can be used independently of OpenAI's model service.

Why BSD, then? [BSD grew out of work on Unix at the University of California, Berkeley](https://docs.freebsd.org/en/books/handbook/introduction/#history), initially tied to AT&T's code and licensing, and eventually became freely redistributable in both source and binary form. Its descendants are useful operating systems in their own right, and their permissive licenses also allow their code to become part of proprietary products (as happened with [FreeBSD](https://www.freebsd.org/) [code](https://www.playstation.com/en-us/oss/ps4/freebsd-kernel/) in Sony's [PlayStation 4](https://www.playstation.com/en-us/ps4/)).

The BSD part is that someone can take the code and build their own product from it, including a proprietary one. OpenAI develops Codex CLI for its own offering, but others can modify and reuse it without having to release their changes as open source.

Codex has no equivalent of BSD's history of removing encumbered Unix code. Its permissive license does, however, allow a fork to make independence from proprietary services its central goal, rather than an alternative configuration. That would be a change in the project's direction, rather than a legal separation. **A vendor can develop the software without having exclusive control over what others build from it.**

## OpenCode and the like: independence from model providers

[OpenCode](https://opencode.ai/) puts the choice of model provider near the center of its offering. Its [documentation](https://opencode.ai/docs/providers/) covers different model services, local models, and custom endpoints. You can keep the harness and change what it connects to.

The Linux comparison makes sense to me because users can choose between vendors without giving one supplier control over the whole workflow. Commercial interests are very much present (hello, Linux distributions), but users need not depend on any one supplier.

OpenCode does not have to occupy this position alone. [Cline](https://cline.bot/), [Crush](https://charm.land/), and [Maki](https://maki.sh/) fit into the same comparison, with different users preferring different interfaces and ways of working. The variety itself resembles the Linux world: there need not be one product that everyone agrees to use. These harnesses are separate implementations rather than distributions of a shared kernel, but the familiar disagreements about which one suits whom would fit right in.

The BSD and Linux comparisons overlap: OpenCode can also be permissively reused, and Codex CLI supports alternative providers. They emphasize different things, though. The BSD parallel concerns what others can build from the implementation; the Linux parallel concerns users choosing between suppliers. Codex is closely associated with OpenAI, while OpenCode presents provider independence as a reason to choose it.

Supporting multiple model providers does not, by itself, tell us how a project is governed or how users can influence its direction. Provider choice does, however, let you keep the harness while changing the model. For someone who wants to keep using a familiar workflow while changing suppliers, that is a useful property.

## Antigravity and the Macintosh experience

The Macintosh comparison adds another reason to choose a platform: how well it fits with everything else you already use. [Continuity](https://www.apple.com/uk/macos/continuity/) connects a Mac and an iPhone through features such as Handoff and Universal Clipboard, while [AirPlay](https://support.apple.com/guide/iphone/intro-to-continuity-iphf5fa30b66/ios) extends the picture to Apple TV. Each connection gives someone who already owns one Apple device another reason to choose the next one from Apple too. **The integration is itself a reason to choose the product.**

[Antigravity](https://www.antigravity.google/docs/ide/overview/) brings the editor, terminal, browser, and [Agent Manager](https://antigravity.google/product/antigravity-ide) into one development environment. Its competitors offer similar capabilities, so their presence alone does not distinguish Google's approach. The Macintosh comparison concerns how well Google brings them together with the services a team already uses.

The connections extend to [Gmail, Drive, Docs, Sheets, Slides, Calendar, and Chat](https://codelabs.developers.google.com/google-workspace-mcp-antigravity) through Google's Workspace MCP servers. These currently require Developer Preview access, configuration, and OAuth authorization; using a Gemini model alone does not provide that access. For a team already using Workspace, a request to change a web page could begin in an email, refer to a specification in Docs, and finish with code changes and an email draft reporting the result. Google's [documentation](https://developers.google.com/workspace/guides/configure-mcp-servers) describes the tools for reading that data and taking actions, although a particular workflow still depends on permissions and agent behavior.

This approach can include open-source components. Apple has [Darwin, WebKit, and Swift](https://opensource.apple.com/); Google publishes the code for Antigravity's [Python SDK](https://github.com/google-antigravity/antigravity-sdk-python) under [Apache 2.0](https://github.com/google-antigravity/antigravity-sdk-python/blob/main/LICENSE), although running it also requires a separately distributed compiled runtime. Neither example makes the complete system open source.

There is a useful limit to the comparison: competing harnesses, including Claude Code, can use those same MCP servers. A team can keep its Google services while changing its harness. **The Macintosh-like bet is that the complete experience is convenient enough to make people choose it**, even when the individual parts have alternatives. First-party integrations can make staying attractive, while shared protocols can make leaving easier.

## What exactly is open?

Taken together, these examples show why a simple division into open and closed harnesses is insufficient. There are several freedoms involved, and having one does not automatically give us the others:

- The freedom to inspect, modify, and redistribute the harness itself.
- The freedom to write extensions and distribute them independently.
- The freedom to replace the model provider.
- The freedom to connect tools through shared protocols.
- The ability to participate in decisions about the project's direction.
- The ability to move instructions, skills, and workflows to another harness.

For example, a proprietary harness can support independently distributed extensions. An open-source harness can depend heavily on one company's services. A permissively licensed harness can become the basis for a proprietary product. These combinations should hardly surprise anyone familiar with the history of free and open-source software.

When I [wrote about Microsoft's relationship with open source](2016-01-30-i-am-still-not-buying-the-new-open-source-friendly-microsoft-narrative.md), I argued that opening individual components was only part of the issue. File formats, compatibility, and control over the surrounding platform mattered too. I think the same applies here. Having the source code is valuable, but we should also ask how much of our work we can take with us if we decide to leave.

## Where will the lock-in live?

Saying “I use Claude” leaves out part of the setup. Claude through Claude Code and Claude through OpenCode can have different tools, instructions, permissions, and ways of presenting results. Those instructions include built-in instructions supplied by the harness, as well as instructions supplied by the user or repository. Even with the same model, they can change how an agent approaches a task, uses tools, and decides when to ask for help. The model is only one part of the system doing the work. The harness determines which tools the model can request, how those requests are authorized and executed, and which results return to the model as context for its next step.

This makes models somewhat like processors and agent harnesses somewhat like operating systems. Changing the model can change behavior far more than replacing a processor with a compatible one, so I would not push that analogy too far. Still, it helps explain why choosing a model does not settle the choice of development environment.

Suppose a team has spent months building reusable skills, writing repository instructions, connecting internal tools, and agreeing on how to review agent-generated changes. Changing a model setting might take a minute. Moving all of that to a different harness and checking that it still works as expected could take considerably longer.

The model could become easier to replace while the workflow becomes harder to move. Some of that is ordinary migration work, which also exists between independent open-source tools. It becomes vendor lock-in when essential integrations or accumulated context remain tied to the original supplier and cannot readily be transferred.

Support for existing conventions reduces that migration work. A harness that can use the instructions and tools a team already has is easier to adopt. The source license matters, but so does the amount of work required to switch. The same compatibility question came up when [considering ProperDocs and Zensical as alternatives to MkDocs](2026-04-14-the-future-of-mkdocs-properdocs-and-zensical.md) for building this website.

## MCP and the role of shared standards

MCP's [documentation](https://modelcontextprotocol.io/docs/getting-started/intro) compares it to a USB-C port: a shared connection lets the same tools and data sources work with different AI applications. That gives competing harnesses a common set of services to work with.

MCP does not make two agents behave identically. They can connect to the same server and still use its tools differently, apply different permission rules, or produce different results. The practical benefit is narrower: a team can reuse the connection instead of rebuilding it for each harness.

This reminds me of [the role of web standards in the browser wars](2015-05-01-browser-wars.md). Being able to access the same websites through different browsers gave users a choice and gave new browser implementations a chance. Shared interfaces between agents and tools could help in a similar way.

Open protocols can also make proprietary software more attractive. If a harness works with tools I already use elsewhere, adopting it becomes easier. Supporting MCP does not make the harness open source, just as supporting Internet protocols did not make Windows open source.

## How far does the comparison go?

At some point, assigning an operating system to every harness stops being useful. [Cursor](https://cursor.com/), for example, suggests another possibility: the editor remains the center of development and incorporates agents into that workflow. Meanwhile, harnesses gain graphical interfaces and model vendors offer complete development environments. These approaches can overlap, and I expect the boundaries to keep moving.

Cloud platforms offer another useful comparison, especially when a harness depends on hosted services and integrations that are difficult to reproduce elsewhere. The operating-system comparison highlights compatibility, code reuse, and the tools and conventions built around an implementation; the cloud comparison draws attention to service dependencies and the cost of moving. For agents embedded in enterprise platforms, that second comparison may be more useful.

There are also some welcome differences from the old operating-system wars. Trying another harness does not require repartitioning a disk, and several agents can work on the same repository. Instructions and tools can sometimes be reused with little effort. As teams build more elaborate workflows, keeping that ability to move will matter more.

The four comparisons describe ways a development tool can become a platform: other tools follow its conventions, other products reuse its code, users can choose between suppliers, or its integrated experience attracts users. A successful harness could combine several of these approaches.

For me, the useful lesson from the operating-system wars is to consider both what attracts us to a platform and what would keep us there. I want tools that work well today, but I also want to keep my instructions, integrations, and work if I decide to switch tomorrow.
