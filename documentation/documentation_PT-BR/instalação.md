# Instalação do PyVAWT

Este documento explica como instalar o **PyVAWT**, um simulador de turbinas eólicas verticais com base no modelo de cilindro atuador.

---

## Requisitos

- Python **3.13+**  
- Git  
- Sistema operacional: **Linux**, **macOS** ou **Windows**

---

## Instalação com `uv` (recomendada)

O `uv` é um gerenciador de pacotes moderno e extremamente rápido, compatível com o padrão `pyproject.toml`.

### 1. Instale o `uv` (caso ainda não tenha)

```bash
# macOS / Linux
curl -Ls https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex
```

### 2. Clone este repositório

```bash
git clone https://github.com/SEU_USUARIO/pyvawt.git
cd pyvawt
```

> Substitua `SEU_USUARIO` pelo seu nome de usuário ou organização no GitHub, se aplicável.

### 3. Instale as dependências do projeto

```
# Instalação normal
uv pip install .

# Modo editável (para desenvolvimento)
uv pip install -e .
```

## ▶️ Como rodar o projeto

1. Edite o arquivo de configuração (`config.yaml`) na raiz do projeto.

2. Execute o script principal:

```bash
python src/pyvawt/main.py
```
Ou rode um dos estudos de caso da pasta `examples:`

```
python examples/run_bianchini2018.py
```
## Dependências principais

O PyVAWT utiliza as seguintes bibliotecas:

- **NumPy** – vetores e matrizes  
- **SciPy** – funções científicas  
- **Matplotlib** – geração de gráficos  
- **h5py** – leitura de arquivos HDF5  
- **Pandas** – manipulação de dados  
- **AeroSandbox** – perfis aerodinâmicos  
- **Typing** – anotações de tipo (já incluído no Python 3.13+)

---

## Dicas para Linux

Se estiver usando Linux, certifique-se de que as bibliotecas de desenvolvimento do HDF5 estão instaladas:

```bash
sudo apt-get update
sudo apt-get install libhdf5-dev
```
