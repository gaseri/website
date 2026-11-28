document.addEventListener("click", (event) => {
  const button = event.target.closest(".openstreetmap-load")
  if (!button) {
    return
  }

  const map = button.closest(".openstreetmap-embed").querySelector("iframe")
  map.src = map.dataset.src
  map.hidden = false
  button.remove()
  map.focus()
})
