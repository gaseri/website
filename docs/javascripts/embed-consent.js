document.addEventListener("click", (event) => {
  if (event.target.closest(".embed-cookie-settings")) {
    window.location.hash = "__consent"
  }
})

function processInstagramEmbeds() {
  if (window.instgrm?.Embeds) {
    window.instgrm.Embeds.process()
    return
  }

  if (document.getElementById("instagram-embed-script")) {
    return
  }

  const script = document.createElement("script")
  script.id = "instagram-embed-script"
  script.src = "https://www.instagram.com/embed.js"
  script.addEventListener("load", () => window.instgrm?.Embeds?.process(), { once: true })
  script.addEventListener("error", () => script.remove(), { once: true })
  document.head.append(script)
}

document$.subscribe(() => {
  const consent = __md_get("__consent")

  for (const embed of document.querySelectorAll("[data-embed-consent]")) {
    if (consent?.[embed.dataset.embedConsent] !== true) {
      continue
    }

    const template = embed.querySelector("template")
    if (template) {
      template.replaceWith(template.content.cloneNode(true))
    } else {
      const frame = embed.querySelector("iframe[data-src]")
      frame.src = frame.dataset.src
      frame.removeAttribute("data-src")
      frame.hidden = false
    }

    embed.querySelector("[data-embed-placeholder]").hidden = true
    embed.removeAttribute("data-embed-consent")
  }

  if (consent?.instagram === true && document.querySelector(".instagram-media")) {
    processInstagramEmbeds()
  }
})
