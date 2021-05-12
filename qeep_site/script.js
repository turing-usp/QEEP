console.log("Script rodando");

const pokefile = document.getElementById("pokefile");
const pokeinput = document.getElementById("image-input");

pokefile.parentElement.onclick = () => pokefile.click();
pokefile.onchange = (e) => {
  const [file] = pokefile.files;
  if (file) {
    pokeinput.src = URL.createObjectURL(file);
    pokeinput.style.display = "block";
    pokeinput.previousElementSibling.style.display = "none";
  }
};
