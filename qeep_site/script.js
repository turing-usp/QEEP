console.log("Script rodando");

const QEEP_URL = "https://mcdxkzn708.execute-api.sa-east-1.amazonaws.com/qeep"

const pokefile = document.getElementById("pokefile");
const pokeinput = document.getElementById("image-input");

pokefile.parentElement.onclick = () => pokefile.click();

const showInputFile = (file) => {
    pokeinput.src = URL.createObjectURL(file);
    pokeinput.style.display = "block";
    pokeinput.previousElementSibling.style.display = "none";
}

const uploadFile = (file) => {
  console.log(`sending: ${file}`)
  return fetch(QEEP_URL, {
    method: 'POST',
     headers: {
      "Content-Type": file.type,
     },
      body: file
  }).then(
    response => response.json()
  ).then(
    success => {
      console.log(success);
      return success.body.outputlink;
    }
  ).catch(
    error => console.log(error)
  );
}

pokefile.onchange = (e) => {
  const [file] = pokefile.files;
  if (file) {
    showInputFile(file);
    uploadFile(
      file
    ).then(
      outputlink => console.log("agora é só fazer esperar até o ", outputlink)
    );
  }
};
