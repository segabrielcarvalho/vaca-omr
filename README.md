# VACA OMR - API de Reconhecimento Óptico de Marcas

Uma API FastAPI para reconhecimento óptico de marcas (OMR - Optical Mark Recognition) em folhas de resposta, utilizando marcadores ArUco para detecção e alinhamento automático.

## 📋 Sobre o Projeto

O VACA OMR é um sistema de processamento de imagens desenvolvido para reconhecer automaticamente respostas marcadas em folhas de prova ou questionários. O sistema utiliza técnicas de visão computacional com OpenCV e marcadores ArUco para garantir alta precisão no reconhecimento das marcações.

### Principais Funcionalidades

- **Detecção automática de folhas** usando marcadores ArUco
- **Correção de perspectiva** automática da imagem
- **Reconhecimento de marcações** em questões de múltipla escolha (A, B, C, D, E)
- **Detecção de respostas ambíguas** (múltiplas marcações)
- **API REST** para integração com outros sistemas
- **Processamento em tempo real** de imagens enviadas

## 🛠️ Tecnologias Utilizadas

- **Python 3.12**
- **FastAPI** - Framework web moderno e rápido
- **OpenCV** - Biblioteca de visão computacional
- **NumPy** - Processamento numérico
- **Uvicorn/Gunicorn** - Servidor ASGI
- **Docker** - Containerização

## 🚀 Instalação e Execução

### Pré-requisitos

- Docker


#### `POST /api/v1/omr`
Processa uma imagem enviada via multipart/form-data.

**Parâmetros:**
- `file` (FormData): Arquivo de imagem
- `threshold` (float, opcional): Limiar de detecção (padrão: 0.50)
- `delta` (float, opcional): Margem de erro (padrão: 0.12)

**Exemplo de uso:**
```bash
curl -X POST "http://localhost:11001/api/v1/omr" \
     -F "file=@folha_resposta.jpg" \
     -F "threshold=0.50" \
     -F "delta=0.12"
```

#### `POST /api/v1/omr-bytes`
Processa dados de imagem enviados como bytes no corpo da requisição.

**Parâmetros de consulta:**
- `threshold` (float, opcional): Limiar de detecção (padrão: 0.50)
- `delta` (float, opcional): Margem de erro (padrão: 0.12)

#### `GET /api/v2/health`
Healthcheck da versão dinâmica do motor OMR.

**Resposta:**
- `status`
- `service`
- `engineVersion`

#### `POST /api/v2/omr/process`
Processa folha usando geometria dinâmica (`template-driven`).

**Body JSON:**
- `imageBase64` (string, obrigatório)
- `compiledGeometryJson` (objeto JSON ou string JSON, obrigatório)
- `threshold` (float, opcional): padrão `0.50`
- `delta` (float, opcional): padrão `0.12`

**Resposta de sucesso:**
- `engineVersion`
- `qr`
- `registration`
- `answers`
- `answers_numeric`
- `timings`
- `images.rectifiedBase64`
- `images.overlayBase64`

**Resposta de erro:**
- `success = false`
- `error.code`
- `error.message`
- `timings`

### Códigos de Resposta

- `200` - Sucesso no processamento
- `400` - Erro na requisição (arquivo inválido, formato incorreto)
- `500` - Erro interno do servidor

## 🔧 Configuração

### Variáveis de Ambiente

- `PORT`: Porta do servidor (padrão: 11001)
- `OMP_NUM_THREADS`: Número de threads OpenMP (padrão: 1)
- `OPENCV_OPENCL_RUNTIME`: Runtime OpenCL (padrão: disabled)

### Parâmetros de Processamento

- **threshold**: Limiar para detecção de marcações (0.0 - 1.0)
- **delta**: Margem de tolerância para ambiguidade (0.0 - 1.0)

## 📖 Como Funciona

### 1. Detecção da Folha
O sistema usa marcadores ArUco (IDs 0, 1, 2, 3) nos cantos da folha para:
- Detectar automaticamente a área da folha
- Corrigir a perspectiva da imagem
- Padronizar o tamanho (1400x1000 pixels)

### 2. Localização das Questões
- Calcula posições predefinidas das áreas de marcação
- Mapeia cada questão para suas respectivas alternativas (A-E)

### 3. Análise das Marcações
- Analisa a intensidade dos pixels em cada área de marcação
- Compara com o limiar definido para determinar se está marcado
- Detecta marcações ambíguas (múltiplas alternativas marcadas)

### 4. Resultado
- Converte as detecções em respostas legíveis
- Fornece estatísticas detalhadas do processamento
- Retorna dados estruturados em JSON

## 🔍 Estrutura do Projeto

```
src/
├── main.py                 # Aplicação FastAPI principal
├── routes/
│   ├── routes_v1.py        # Endpoints da API v1 (legado)
│   └── routes_v2.py        # Endpoints da API v2 (dinâmico)
├── services/
│   └── omr/
│       ├── __init__.py
│       ├── omr_service.py    # Lógica principal do OMR v1
│       └── omr_service_v2.py # Lógica template-driven do OMR v2
└── utils/
    ├── aruco_detector.py   # Detecção de marcadores ArUco
    ├── choice_detector.py  # Detecção de marcações
    ├── file_handler.py     # Manipulação de arquivos
    ├── sheet_geometry.py   # Geometria da folha
    └── visualization.py    # Visualização e debug
```

## ⚡ Performance

- Processamento típico: ~20 ms por imagem
- Suporte a imagens de alta resolução
- Otimizado para execução em containers Docker
- Baixo uso de memória com processamento sequencial

---

**Projeto desenvolvido como parte do TCC - Trabalho de Conclusão de Curso**
