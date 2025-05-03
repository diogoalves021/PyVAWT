# PyVAWT

PyVAWT é um programa em Python para simular turbinas eólicas de eixo vertical (VAWTs) utilizando o método do cilindro atuador. Esse método modela o comportamento aerodinâmico de VAWTs representando o rotor como uma série de cilindros que interagem com o escoamento do vento.

## Descrição

O objetivo do projeto é fornecer insights sobre o desempenho de turbinas eólicas, sendo ideal para pesquisadores e engenheiros envolvidos na otimização e análise de performance de turbinas.

**Status atual:**  
O projeto está em desenvolvimento ativo. Atualmente, funcionalidades básicas já foram implementadas, incluindo:

1. Simulação de uma turbina
2. Simulação de duas turbinas
3. Modelo simples de esteira (wake)
4. Leitura de parâmetros da simulação a partir de um arquivo `.json`

> **Atenção:** Este projeto ainda está em fase inicial de desenvolvimento. Não é recomendado para análises de engenharia críticas, pois a precisão e funcionalidades estão sujeitas a mudanças significativas.

---

## Funcionalidades

### 1. Geração de Dados Aerodinâmicos (`data_generation/generator.py`)

Gera curvas de sustentação (Cl) e arrasto (Cd) utilizando o módulo `neuralfoil` da biblioteca [AeroSandbox](https://aerosandbox.com/), a partir das condições de simulação fornecidas.

- Escolha de qualquer perfil NACA
- Definição dos números de Reynolds e Mach
- Possibilidade de simular múltiplos perfis de forma automática

**✓ Alta precisão**  
**✗ Tempo de geração alto (~10 minutos por perfil)**

Ideal para análises detalhadas e confiáveis de desempenho aerodinâmico.

---

### 2. Leitura de Dados Pré-Gerados (`data_reading/reader.py`)

Importa arquivos com curvas Cl/Cd obtidos por outras ferramentas (como XFoil, QBlade ou o próprio AeroSandbox) e interpola os dados para uso na simulação.

- Suporte a múltiplos formatos de entrada
- Simulação rápida (tipicamente ~5 segundos)

**✓ Alta velocidade de simulação**  
**✗ Menor precisão (dependente da qualidade dos dados de entrada)**

Ideal para testes rápidos, estudos paramétricos ou otimizações.

---

## Uso

Crie um arquivo de configuração `config.json` com os parâmetros da simulação:

```json
{
    "turbine": {
        "r": 17.5,
        "twist": 0.0,
        "delta": 0.0,
        "chord": 1.75,
        "B": 2,
        "solidity": 0.1,
        "centerX": 0,
        "centerY": 0,
        "Omega": 0.0,
        "ntheta": 36
    },
    "environment": {
        "Vinf": 1.0,
        "rho": 1.225,
        "mu": 1.7894e-5
    },
    "simulation": {
        "var_omega_vinf": 0,
        "num_turbines": 2,
        "aero_profile": "data/NACA_0012_mod.dat"
    }
}


After setting up your environment and configuring the simulation parameters, you can run the code using the following command:

```bash
python3 -m examples.Solidity_chord_vinf_Fallstudie
```

This will start the simulation, and the results will be displayed according to the parameters specified in the JSON configuration file, and in this case, according to the aerodynamic profiles you select in the code. The results will be saved in folders with the name of the selected parameters. In these folders there will be a json file with the parameters used in the simulation, an .eps file with the data graph and a .dat file with the data.

## Directories Structure

```

project_root/
│
├── src/                       # Main application code
│   ├── __init__.py
│   ├── main.py                # Main entry point
│   ├── config/                # Static configuration files
│   │   └── config.json
│   ├── data_generation/       # Cl/Cd data generation
│   │   ├── __init__.py
│   │   └── generator.py
│   ├── data_reading/          # Cl/Cd data reading and interpolation
│   │   ├── __init__.py
│   │   └── reader.py
│   ├── simulation/            # Simulation core logic
│   │   ├── __init__.py
│   │   └── simulator.py
│   └── utils/                 # Reusable utility functions
│       ├── __init__.py
│       └── helpers.py
│
├── tests/                     # Unit and functional tests
│   ├── __init__.py
│   ├── test_generator.py
│   ├── test_reader.py
│   └── test_simulation.py
│
├── examples/                  # Usage examples and case studies
│   └── example_case_1/
│       ├── params.json
│       └── run_case.py
│
├── results/                   # Simulation results (organized by case/date)
│   └── case_01/
│       └── results.json
│
├── data/                      # Raw data or airfoil definitions
│   └── airfoil_001.dat
│
├── README.md
├── requirements.txt
└── setup.py                   # Package installation file

```

## Dependecies

The dependencies for this project are listed in the `pyproject.toml` file. To install the required dependencies, you can use [Poetry](https://python-poetry.org/) or [pip](https://pip.pypa.io/en/stable/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.