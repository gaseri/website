document.addEventListener("click", (event) => {
  if (event.target.closest(".youtube-cookie-settings")) {
    window.location.hash = "__consent"
  }
})

document$.subscribe(() => {
  if (__md_get("__consent")?.youtube !== true) {
    return
  }

  for (const player of document.querySelectorAll(".youtube-embed iframe[data-src]")) {
    player.src = player.dataset.src
    player.removeAttribute("data-src")
    player.hidden = false
    player.parentElement.querySelector("p").hidden = true
  }
})
