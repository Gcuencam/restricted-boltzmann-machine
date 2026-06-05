# RBM para recomendar platos — casos interesantes para el artículo

Experimento: una **Restricted Boltzmann Machine** aprende, sin etiquetas, los
gustos latentes de los clientes de una empresa de comida a domicilio. El objetivo
es demostrar que descubre **un gusto que no se ve a simple vista: el picante**.

## El montaje

12 platos repartidos en 4 cocinas, más un eje transversal (picante):

| Cocina | Platos | Picante 🌶️ |
|---|---|---|
| mexicano | Tacos al pastor, Quesadillas, **Totopos con jalapeños** | Totopos |
| italiano | Rigatoni Carbonara, Lasaña, Risotto | *(ninguno — cocina de control)* |
| cuchara | Fabada, **Lentejas con chorizo picante**, Sopas de ajo | Lentejas |
| asiático | **Kimchi Chigae**, Sushi, **Pad Thai** | Kimchi, Pad Thai |

Cada cliente sintético tiene **dos rasgos latentes independientes**: una `cocina`
y si es `picante` o no. Un amante del picante eleva su probabilidad en **todos**
los platos picantes, sin importar de qué cocina sean. Ese cruce es lo que la red
debe redescubrir. La cocina italiana no tiene plato picante propio: sirve de
control para demostrar que el picante es un eje ortogonal y no "una cocina más".

## Cómo reproducirlo

Los tres pasos, en orden. Cada comando se muestra con **todos sus parámetros
explícitos** (coinciden con los valores por defecto del código):

```bash
# 1. Generar los datos sintéticos (deterministas: seeds fijadas en el código)
npm run generate
#    → data/train.json      240 usuarios (4 cocinas × 60), seed 42, ruido 0.03
#    → data/new-users.json  10 usuarios de demostración,   seed 22, ruido 0

# 2. Entrenar la RBM
npx tsx src/train.ts 6 500 0.1 32 42 0.02
#                    │ │   │   │  │  └─ hiddenBiasDecay : decaimiento de los bias ocultos
#                    │ │   │   │  └──── seed            : semilla (init + barajado + muestreo CD-1)
#                    │ │   │   └─────── batchSize       : tamaño de mini-batch
#                    │ │   └─────────── learningRate    : tasa de aprendizaje
#                    │ └─────────────── epochs          : número de epochs
#                    └───────────────── nHidden         : nº de unidades ocultas
#    (equivale a `npm run train`, que usa exactamente estos defaults)

# 3. Inferencia + tablas (matriz W, preferencias, activaciones ocultas)
npx tsx src/index.ts 6
#                    └─ nHidden : debe coincidir con el modelo entrenado (carga data/model-6.json)
#    (equivale a `npm run dev`)
```

### Parámetros de `train` (posicionales)

| Pos | Parámetro         | Default | Qué hace |
|-----|-------------------|---------|----------|
| 1   | `nHidden`         | `6`     | Unidades ocultas. ≥5 para 4 cocinas + picante. |
| 2   | `epochs`          | `500`   | Pasadas por el dataset. |
| 3   | `learningRate`    | `0.1`   | Tasa de aprendizaje de CD-1. |
| 4   | `batchSize`       | `32`    | Tamaño de mini-batch (SGD). |
| 5   | `seed`            | `42`    | Semilla global → entrenamiento determinista. |
| 6   | `hiddenBiasDecay` | `0.02`  | Decae los bias ocultos para desenredar las cocinas (ver §5). |

### Parámetros de `index` / `dev` (posicionales)

| Pos | Parámetro | Default | Qué hace |
|-----|-----------|---------|----------|
| 1   | `nHidden` | `6`     | Carga `data/model-${nHidden}.json` y dimensiona las tablas. |

El entrenamiento es **totalmente determinista** (init, barajado y muestreo de CD-1
sembrados): dos corridas producen pesos idénticos (mismo md5). Reproducible para el
artículo.

## La estructura que la red descubre sola

Con 6 unidades ocultas, 5 quedan interpretables (H2 es redundante):

```
Hidden 1 = cuchara    Hidden 3 = mexicano    Hidden 5 = italiano
Hidden 4 = asiatico   Hidden 6 = PICANTE 🌶️
```

La unidad del picante (H6) se detecta automáticamente: es la que más pesa sobre los
platos picantes frente al resto (peso picante − no picante = **+2.83**). Sus pesos
son un filtro de manual:

| Plato | Peso en H6 |
|---|---|
| Kimchi Chigae 🌶️ | **+2.46** |
| Totopos con jalapeños 🌶️ | **+2.39** |
| Pad Thai 🌶️ | **+1.40** |
| Lentejas con chorizo picante 🌶️ | **+1.32** |
| Sopas de ajo | −2.08 |
| Fabada | −2.81 |

Positivo en los 4 platos picantes (de 3 cocinas distintas), negativo en el resto.
**Nadie etiquetó "picante" en los datos** — la red lo infiere de las co-ocurrencias.

## Casos interesantes para contar

### 1. El gusto latente, descubierto
La unidad H6 se enciende para los amantes del picante y se apaga para el resto,
**con independencia de la cocina**. Es el corazón del artículo: un eje de gusto que
no aparece como ninguna columna de los datos, reconstruido como factor latente.

### 2. Pares emparejados: solo el picante los separa
Dos comensales de la **misma cocina** a los que únicamente la unidad del picante
distingue:

| | no picante | picante 🌶️ | H6 (no) → (sí) |
|---|---|---|---|
| mexicano | Arancha | Maria | 0.47 → **1.00** |
| italiano | Antoni | Marta | 0.36 → **0.72** |
| cuchara | Juan | Ramón | 0.00 → **0.67** |
| asiático | Elena | Laura | 0.15 → **1.00** |

Mismos platos de cocina, distinta firma en H6. El picante es lo único que cambia.

### 3. Marta, la prueba de fuego (cocina italiana + picante)
Marta pide cocina **italiana**, que **no tiene ningún plato picante propio**. Aun
así la red la marca como picante (H6 = **0.72**) porque cruza a otras cocinas:
pide Lentejas, Kimchi… Demuestra que el picante es verdaderamente transversal y no
un subproducto de ninguna cocina concreta. Es el caso más difícil y la red lo
resuelve bien.

### 4. Varios factores a la vez
La última columna ("Factores más activos") lista **todas** las unidades que se
encienden (P > 0.5) en cada cliente, ordenadas de mayor a menor y coloreadas en
terminal por intensidad: 🟢 verde ≥0.90, 🟡 amarillo ≥0.75, 🟠 naranja ≥0.50.
Recordatorio del mapa: H1=cuchara, H3=mexicano, H4=asiatico, H5=italiano, H6=picante.

Los amantes del picante encienden **su cocina Y el picante** a la vez:

```
Maria   mexicano  Sí  → H3 H6        (mexicano + picante)
Laura   asiatico  Sí  → H4 H6 H1     (asiatico + picante + …)
Jesús   asiatico  Sí  → H4 H6 H1     (asiatico + picante + …)
Ramón   cuchara   Sí  → H1 H3 H6     (cuchara + … + picante)
```

Un cliente no es "una cosa": es una **combinación** de factores latentes. Para
recomendar, la red puede sugerir platos picantes de OTRA cocina a un Jesús (asiático
picante) — exactamente el tipo de recomendación no obvia que justifica el sistema.

### 5. La lección técnica: por qué casi no funciona
Al principio las cocinas **no** se separaban: el `argmax` del factor dominante daba
resultados aleatorios. El culpable no era el modelo sino los **bias ocultos
saturados**: algunas unidades desarrollaban un bias enorme (bh ≈ +5.4), quedaban
"encendidas por defecto" (σ(5.4) ≈ 0.996 para todos) y su activación dejaba de
significar nada. La solución fue **decaer los bias ocultos** (`hiddenBiasDecay`):
con eso `max|bh|` baja de ~5.8 a ~1.5, las activaciones vuelven a estar gobernadas
por la evidencia y cada cocina cae en su propia unidad. Buena moraleja para el
artículo: en modelos no supervisados, **interpretar** las unidades es la mitad del
trabajo.

## Resultados finales

- **Cocina: 10/10** clientes asignados a su cocina correcta.
- **Picante: 10/10** clasificados correctamente (no picantes < 0.5, picantes > 0.5).
- Bijección perfecta cocina ↔ unidad oculta.
- Entrenamiento determinista y reproducible.

## Salvedades honestas (para no sobrevender)

- El desenredo de cocina llega al 100% en los 10 clientes de demostración; sobre el
  set de entrenamiento completo (240) la pureza es ~0.82, no perfecta. Las RBM no
  garantizan unidades interpretables una-a-una; aquí ayudó el bias decay.
- El bias decay suaviza las probabilidades, así que el picante separa con menos
  margen que sin él (0.47/0.72 en vez de 0.01/0.99), aunque sigue acertando 10/10.
- Los datos son sintéticos y diseñados para que el experimento sea limpio. Es una
  demostración pedagógica, no una validación sobre datos reales.
