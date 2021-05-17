console.log("Script rodando");

const QEEP_URL = "https://mcdxkzn708.execute-api.sa-east-1.amazonaws.com/qeep"

const pokefile = document.getElementById("pokefile");
const pokeinput = document.getElementById("image-input");
const pokeoutput = document.getElementById("image-output");
const pokeoutputloading = document.getElementById("output-loading");

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

const waitFileShow = (src) => {
    pokeoutput.previousElementSibling.style.display = "none"; // tira a imagem da camera
    pokeoutputloading.style.display = "block"; // mostra carregamento

    const tryUpdateImage = new Promise((resolve, reject) => {
      const interval = setInterval(() => {
        fetch(src)
          .then((res) => {
            console.log(res.status);
            if (res.status === 404) {
              console.log("esperando o ", src)
              return;
            }
            clearInterval(interval);
            resolve(res);
          })
      }, 30 * 1000);
    });
    tryUpdateImage.then(() => {
      console.log("predição realizada com sucesso");
      pokeoutput.style.display = "block";
      pokeoutputloading.style.display = "none"; // mostra carregamento
      pokeoutput.src = src;
    });
}

pokefile.onchange = (e) => {
  const [file] = pokefile.files;
  if (file) {
    if (file.length > 16384) {
      alert("Arquivo muito grande, por favor selecione arquivos menores que 16MB");
      return;
    }
    showInputFile(file);
    uploadFile(file).then(waitFileShow);
  }
};
