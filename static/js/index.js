function copyBibtex() {
  const text = document.getElementById('bibtex-content').innerText;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('copy-btn');
    const original = btn.innerText;
    btn.innerText = 'Copied!';
    setTimeout(() => { btn.innerText = original; }, 1500);
  });
}
