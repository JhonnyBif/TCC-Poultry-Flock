# Guia de Instalação e Execução — Projeto TCC: VISÃO COMPUTACIONAL PARA IDENTIFICAR AGLOMERAÇÃO DE AVES CONFINADAS

Este documento explica, passo a passo, como configurar e executar o projeto de VISÃO COMPUTACIONAL PARA IDENTIFICAR AGLOMERAÇÃO DE AVES CONFINADAS. Siga cuidadosamente cada etapa, na ordem apresentada.  
Não pule nenhum passo.

---

## 1. Baixar o repositório

Acesse o link abaixo:

**Repositório GitHub:**  
[https://github.com/JhonnyBif/TCC-Poultry-Flock](https://github.com/JhonnyBif/TCC-Poultry-Flock)

Na página, clique no botão **Code** (botão verde) e depois em **Download ZIP**.  
Após o download, **extraia o arquivo ZIP** (botão direito → “Extrair aqui” ou “Extrair tudo”).  
Lembre-se de onde a pasta foi salva — você vai precisar acessá-la nas próximas etapas.

---

## 2. Instalar as ferramentas necessárias

Antes de executar o projeto, será necessário instalar alguns programas.  
Abaixo estão os links para download e vídeos explicativos.  
Assista aos vídeos e siga exatamente o que é mostrado.

### 2.1. Instalar Node.js

O Node.js é necessário para rodar a parte do projeto que exibe a interface (frontend).

**Instalação:**  
[Baixar Node.js](https://nodejs.org)

**Tutorial em vídeo:**  
[Como instalar Node.js (YouTube)](https://www.youtube.com/watch?v=La6kH33-AVM&pp=ygUXaG93IHRvIGluc3RhbGwgbm9kZSB2MjDSBwkJAwoBhyohjO8%3D)

---

### 2.2. Instalar Angular 17.3

O Angular é o framework usado no frontend.  
Após instalar o Node.js, abra o terminal e siga o tutorial abaixo para instalar o Angular.

**Tutorial em vídeo:**  
[Como instalar Angular 17.3 (YouTube)](https://www.youtube.com/watch?v=Xu7R_aMz070&pp=ygUbaG93IHRvIGluc3RhbGwgYW5ndWxhciAxNy4z)

---

### 2.3. Instalar Python

O Python é usado para processar vídeos e imagens (backend do projeto).

**Download:**  
[Baixar Python](https://www.python.org/downloads/)

**Tutorial em vídeo:**  
[Como instalar Python (YouTube)](https://www.youtube.com/watch?v=YKSpANU8jPE&pp=ygUUaG93IHRvIGluc3RhbGxweXRob24%3D)

Durante a instalação, marque a opção **“Add Python to PATH”** antes de clicar em “Install Now”.

---

### 2.4. Instalar Pip

O Pip é o gerenciador de pacotes do Python, usado para instalar as bibliotecas necessárias.  
Em geral, ele é instalado junto com o Python.  
Caso não esteja, siga o vídeo abaixo:

**Tutorial em vídeo:**  
[Como instalar Pip (YouTube)](https://www.youtube.com/watch?v=oyjOBhwSl-8&pp=ygUSaG93IHRvIGluc3RhbGwgcGlw0gcJCQMKAYcqIYzv)

---

### 2.5. Instalar Visual Studio Code (VS Code)

O VS Code é o editor onde você abrirá o projeto e executará os comandos.

**Download:**  
[Baixar VS Code](https://code.visualstudio.com/download)

**Tutorial em vídeo:**  
[Como instalar VS Code (YouTube)](https://www.youtube.com/watch?v=cu_ykIfBprI&pp=ygUVaG93IHRvIGluc3RhbGwgdnNjb2Rl)

Após instalado, abra o VS Code.  
Você pode abrir o terminal do próprio VS Code clicando em **Terminal → Novo Terminal**.

---

## 3. Executar o Frontend

O frontend é a parte visual do sistema (interface que o usuário vê no navegador).

1. Abra o VS Code ou o terminal.
2. Acesse a pasta do projeto chamada **monitoramento-frangos-front**.  
   Exemplo de comando (caso a pasta esteja na área de trabalho):

   ```bash
   cd Desktop/TCC-Poultry-Flock/monitoramento-frangos-front
   ```

3. No terminal, execute o comando abaixo para instalar as dependências:

   ```bash
   npm i
   ```

   Esse processo pode demorar alguns minutos.

4. Após a instalação, execute o comando:

   ```bash
   ng s
   ```
   ou, alternativamente:
   ```bash
   npm start
   ```

   Se tudo estiver correto, aparecerá uma mensagem “Compiled successfully” e o sistema estará acessível no navegador no endereço:  
   [http://localhost:4200](http://localhost:4200)

Deixe este terminal aberto. Ele precisa continuar rodando.

---

## 4. Executar o Backend

O backend é a parte do sistema que realiza o processamento das imagens e vídeos usando inteligência artificial.

1. Abra outro terminal (ou uma nova janela do VS Code).  
2. Acesse a pasta do backend:
   ```bash
   cd Desktop/TCC-Poultry-Flock/monitoramento_frangos_back
   ```
3. Execute o comando abaixo para instalar as dependências do Python:
   ```bash
   pip install -r requirements.txt
   ```

   Esse arquivo contém todas as bibliotecas necessárias.  
   Ele já está configurado para funcionar em um computador com processador **Ryzen 7** e placa de vídeo **NVIDIA 4060 Ti**.  
   Caso algum erro ocorra, pode ser devido a diferenças de hardware ou drivers.

4. Após a instalação, execute o comando:
   ```bash
   flask run
   ```

   O backend será iniciado e ficará aguardando requisições do frontend.

Deixe este terminal aberto também.

---

## 5. Acessar o Sistema

Com ambos os terminais em execução (frontend e backend), abra o navegador e digite:

```
http://localhost:4200
```

A interface do sistema será exibida.

---

## 6. Realizar o Processamento de Teste

1. Clique no botão **Selecionar Imagem**.  
2. Acesse a pasta do projeto que você baixou anteriormente.  
3. Dentro dela, abra a pasta **imagens** e selecione o arquivo **Crua**.  
4. No sistema, selecione o **radio-button** com a opção **Imagem**.  
5. Clique em **Processar** e aguarde o resultado.

Após o processamento:

- Uma nova imagem chamada **dimensions** será gerada.  
- Também será criado um arquivo de texto chamado **info.txt** na pasta raiz do projeto.  

Abra o arquivo **info.txt**: nele estarão os valores necessários para preencher os campos de dimensões e idade no sistema.

6. Retorne à interface, preencha os valores e clique novamente em **Processar**.  
   Aguarde até o sistema finalizar a análise.

---

## 7. Visualização dos Resultados

Quando o processamento for concluído, será aberto um **modal (janela)** com os resultados.

- Clique na caixa de seleção (select).  
- Ela se expandirá e mostrará as imagens processadas.  
- Clicando em qualquer imagem, ela será ampliada.  
- Abaixo da imagem haverá um botão **Baixar**, que permite salvar a imagem analisada.

---

## 8. Solução de Problemas

Se algo não funcionar corretamente:

1. Verifique se o **Node.js**, **Angular**, **Python** e **Pip** estão instalados.  
2. Feche e abra novamente o VS Code.  
3. Certifique-se de estar nas pastas corretas antes de executar os comandos.  
4. Repita os comandos de instalação, se necessário.  
5. Caso o erro persista, reinicie o computador e tente novamente.  

Se mesmo assim o problema continuar, é possível que esteja relacionado ao hardware (placa de vídeo, drivers, versão de CUDA, etc.). Nesse caso, entre em contato com o responsável pelo projeto para suporte.

---

## 9. Encerramento

Seguindo corretamente todas as etapas, o sistema de **VISÃO COMPUTACIONAL PARA IDENTIFICAR AGLOMERAÇÃO DE AVES CONFINADAS** estará em funcionamento completo, realizando o processamento de imagens e exibindo os resultados de forma automática.  

Mantenha os dois terminais (frontend e backend) abertos enquanto o sistema estiver em uso.  
Para encerrar, pressione **Ctrl + C** em cada terminal e feche o VS Code.
