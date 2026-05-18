# Okoliš AI — Arhitektura sustava

## Što je Okoliš AI?

Okoliš AI je sustav koji koristi iPhone LiDAR skenove (.ply datoteke) za rekonstrukciju vanjskih okruženja — dvorišta, fasade zgrada, zidove, teren, ceste — i pretvara ih u strukturirane 3D reprezentacije. Sustav semantički razumije što je što (zid, cesta, tlo, vegetacija, objekt), a korisnik može interaktivno uređivati scenu (produžiti zid, zamijeniti teren, dodati strukture).

---

## Pregled pipeline-a

Cijeli sustav radi u 15 koraka, podijeljenih u 4 faze:

```
iPhone LiDAR (.ply)
       │
       ▼
 ┌─────────────┐
 │ 1. UČITAVANJE │  ← ply_loader.py
 └──────┬──────┘
        ▼
 ┌─────────────────┐
 │ 2. PREDOBRADA    │  ← preprocess.py
 │   - ukloni šum   │
 │   - downsample    │
 │   - normali       │
 │   - gravity align │
 └──────┬──────────┘
        ▼
 ┌──────────────────┐
 │ 3. GEOMETRIJA     │
 │   - tlo (RANSAC)  │  ← ground.py
 │   - ravnine       │  ← planes.py
 │   - klasteri      │  ← clusters.py
 │   - značajke      │  ← features.py
 └──────┬───────────┘
        ▼
 ┌──────────────────┐
 │ 4. ML SEGMENTACIJA│  ← RandLA-Net (model.py)
 │   - tile-based    │  ← inference.py
 │   - softmax probs │
 └──────┬───────────┘
        ▼
 ┌──────────────────┐
 │ 5. FUZIJA         │  ← hybrid.py
 │   geometrija + ML │
 │   → finalne labele│
 └──────┬───────────┘
        ▼
 ┌──────────────────┐
 │ 6. ZIDOVI         │  ← wall_extractor.py
 │   - WallObject    │
 │   - duljina,      │
 │     visina,       │
 │     debljina      │
 └──────┬───────────┘
        ▼
 ┌──────────────────┐
 │ 7. SCENA + EDITOR │  ← scene.py, ops.py
 │   - produlji zid  │
 │   - zamijeni teren│
 │   - dodaj objekt  │
 └──────┬───────────┘
        ▼
 ┌──────────────────┐
 │ 8. VIZUALIZACIJA  │  ← viewer.py
 │   - Open3D viewer │
 │   - boje po klasi │
 └─────────────────┘
```

---

## Faza 1: Učitavanje i predobrada

### ply_loader.py

Učitava .ply datoteke (format koji iPhone LiDAR exportira). Svaka točka ima x, y, z koordinate, a opcionalno i boju (RGB) i normale. Ako nešto nedostaje, gracefully se postave defaultne vrijednosti (bijela boja, nula-normali).

### preprocess.py

Sirovi sken ima šum i previše točaka. Predobrada radi 4 stvari:

1. **Uklanjanje šuma** — statistički (ukloni točke čiji su susjedi previše daleko) + radius filter (ukloni izolirane točke)
2. **Voxel downsample** — dijeli prostor na kockice (voxele) veličine npr. 3-5 cm i zadrži samo jednu točku po kockici. Smanjuje oblak s milijuna na stotine tisuća točaka
3. **Procjena normala** — za svaku točku izračuna normalu površine koristeći k najbližih susjeda (kNN). Normale se orijentiraju prema viewpointu
4. **Gravity alignment** — detektira ravninu tla pomoću RANSAC-a na najnižim točkama i rotira cijeli oblak tako da je Z os okomita na tlo

---

## Faza 2: Geometrijska analiza

### ground.py — Ekstrakcija tla

Dva pristupa:

- **RANSAC** — traži najveću horizontalnu ravninu u oblaku. Brzo i robusno za ravne terene
- **Grid-based** — dijeli XY prostor na ćelije (npr. 1m × 1m), u svakoj ćeliji pronađe najnižu točku, pa 3×3 median smoothing za glatki prijelaz. Radi i za nagnute terene

Također računa **height_above_ground** za svaku točku — koliko je točka iznad tla. Ovo je ključna značajka za ML model.

### planes.py — Ekstrakcija ravnina (zidovi)

Iterativni RANSAC: ponavlja do 20 puta:

1. Nađi ravninu koja ima najviše inlier-a
2. Provjeri je li vertikalna (normala blizu horizontalne)
3. Provjeri minimalnu površinu
4. Ukloni te točke iz oblaka
5. Ponovi na ostatku

Svaka pronađena ravnina postaje `PlaneSegment` s informacijama: indeksi točaka, jednadžba ravnine, normala, površina, vertikalnost, centroid.

### clusters.py — Klasteriranje ostatka

Točke koje nisu tlo ni ravnine (vegetacija, auti, stupovi) se grupiraju DBSCAN algoritmom. DBSCAN traži gusto povezane grupe točaka — ne traži kružne klastere kao k-means, pa može pronaći nepravilne oblike poput stabala ili automobila.

### features.py — Geometrijske značajke segmenata

Za svaki segment (ravninu, klaster, tlo) izračunavaju se značajke pomoću PCA (Principal Component Analysis):

- **Vertikalnost** — koliko je segment uspravan
- **Planarnost** — koliko je ravan (visoka za zidove)
- **Linearnost** — koliko je izdužen (visoka za stupove)
- **Sferičnost** — koliko je okrugao (visoka za grmlje)
- **Raspon visine** — razlika max-min Z
- **AABB** — bounding box
- **OBB osi** — orijentirani bounding box

---

## Faza 3: ML segmentacija

### Taksonomija — 6 klasa

```
0 = unlabeled    (neklasificirano)
1 = ground        (tlo — trava, zemlja)
2 = road          (cesta, pločnik, parking)
3 = wall          (zid, fasada, ograda)
4 = nature        (vegetacija, drveće, grmlje)
5 = object        (stup, auto, klupa, uličnI namještaj)
```

Sve javne datasete (Toronto-3D, SemanticKITTI, BotanicGarden) mapiramo na ovih 6 klasa pomoću `label_maps.py`.

### Unificiran 5-dimenzionalni feature vektor

Svaka točka dobiva 5 značajki:

```
[R, G, B, intensity, height_above_ground]
```

- **R, G, B** — boja, normalizirane na [0, 1]
- **Intensity** — intenzitet laserskog povratnog signala, [0, 1]. iPhone ga NEMA (0), SemanticKITTI ga ima
- **Height above ground** — visina iznad tla u metrima

### Modality dropout — premošćivanje domain gap-a

Problem: iPhone nema intensity, SemanticKITTI nema boju. Ako model nauči ovisiti o jednoj modalnosti, neće raditi na drugom izvoru.

Rješenje: tijekom treninga nasumično zeriramo RGB kanale (30% šanse) ili intensity kanal (30% šanse). Ovako model uči koristiti bilo koju dostupnu informaciju, a ne ovisiti o jednoj.

### RandLA-Net

Odabrani model za segmentaciju. Zašto baš RandLA-Net:

| Kriterij | RandLA-Net | PointNet++ | KPConv |
|----------|-----------|-----------|--------|
| Veliki oblaci (100k+ točaka) | Da | Ne | Srednje |
| Memorija (6GB VRAM) | OK | OK | Previše |
| Mobilni port (iOS) | Lak | Srednje | Težak |
| Toronto-3D mIoU | ~70% | ~65% | ~75% |

RandLA-Net koristi **random sampling** umjesto FPS (Farthest Point Sampling) — ovo je puno brže za velike oblake. Lokalne značajke izvlači pomoću **Local Spatial Encoding** i **Attentive Pooling**.

Arhitektura u kodu:
- `SharedMLP` — dijeljeni FC slojevi
- `LocSE` — Local Spatial Encoding (pozicijsko kodiranje susjeda)
- `AttentivePool` — ponderirana agregacija s naučenim težinama
- `DilatedResBlock` — dva LocSE + AttentivePool s residual vezom
- `RandLANet` — stem → 2 dilated bloka → klasifikacijska glava

### Tiled inference

Veliki oblak (npr. 1M+ točaka) ne stane u GPU odjednom. Rješenje:

1. Podijeli XY prostor na tile-ove (npr. 50×50m) s preklapanjem
2. Svaki tile neovisno propusti kroz model
3. U preklapajućim zonama usrednji softmax vjerojatnosti
4. Rezultat: (N, 6) matrica — za svaku od N točaka, 6 vjerojatnosti

---

## Faza 4: Hibridna fuzija

### hybrid.py

ML sam po sebi daje ~70% točnosti. Geometrija sama daje grubu klasifikaciju. Kombiniranjem oboje dobivamo bolji rezultat.

Za svaki segment (iz Faze 2):

1. **ML glasovi** — usrednji softmax vektore svih točaka u segmentu
2. **Geometrijski prior** — pravila koja korigiraju ML:
   - Horizontalna ravnina NE MOŽE biti zid → smanji wall score
   - Vertikalna ravnina velike površine → pojačaj wall score
   - Mali izolirani klaster → smanji wall score, pojačaj object
   - Visok, tanak, ne-planaran segment → pojačaj nature (drvo)
3. **Finalna labela** = argmax(ML_scores × geo_weights)
4. **Confidence** = top_score × slaganje ML-a i geometrije

---

## Ekstrakcija zidova

### wall_extractor.py

Kad segment dobije labelu "wall", izvlačimo detaljne parametre:

1. **Plane basis** — normala zida definira koordinatni sustav ravnine (u, v, n)
2. **Projekcija** — projiciraj sve točke zida na 2D (u, v) ravninu
3. **PCA smjer** — pronađi glavni smjer zida u 2D
4. **Percentile endpoints** — 5. i 95. percentil duž glavnog smjera → start i end
5. **Visina** — 5. do 95. percentil u V smjeru
6. **Debljina** — twin-plane detekcija: traži drugu ravninu paralelnu unutar 0.05–0.5m

Rezultat je `WallObject` sa: start, end, direction, length, height, thickness, base_z, plane_normal, confidence.

---

## Scena i uređivanje

### scene.py

`Scene` je glavni podatkovni objekt:

```python
@dataclass
class Scene:
    points: ndarray      # (N, 3) — sve XYZ koordinate
    colors: ndarray      # (N, 3) — RGB boje
    segments: List[Segment]  # svi segmenti s labelama
    walls: List[WallObject]  # ekstrahirani zidovi
    synthetic_mask: ndarray   # True za dodane točke
```

Scena ima KD-tree za brze prostorne upite i `point_to_segment` lookup tablicu.

Ključno: sva uređivanja su **funkcionalna**:

```python
new_scene = scene.replace(points=new_points, colors=new_colors)
```

Stari objekt ostaje nepromijenjen → undo/redo je trivijalan (samo zadrži listu starih scena).

### ops.py — Operacije uređivanja

**extend_wall(scene, wall_id, direction, amount)**
- Pronađe zid po ID-u
- Izračuna novi endpoint u zadanom smjeru
- Generira sintetičke točke na površini produženog dijela (dvije velike face + dva end cap-a)
- Označi ih kao synthetic=True
- Spoji u novu scenu

**replace_terrain(scene, polygon, new_height)**
- Definira poligon u XY ravnini (matplotlib.path.Path za point-in-polygon test)
- Ukloni sve ground točke unutar poligona
- Generiraj ravnu mrežu točaka na novoj visini
- Spoji u novu scenu

---

## Dataseti

### Toronto-3D (primarni za smoke test)

Urbani LiDAR — ulice Toronta snimljene mobilnim senzorima. 4 PLY datoteke (L001–L004). Sadrži: x, y, z, RGB, intensity, labels (8 klasa → mapiramo na naših 6).

Split: L001+L003 trening, L004 validacija, L002 test.

### SemanticKITTI (za budućnost)

Automobilski LiDAR (Velodyne) — .bin + .label format. 22 sekvence, 20 klasa → mapiramo na 6. Ima intensity, NEMA RGB.

### BotanicGarden (za budućnost)

Botanički vrt — PLY/PCD format, 5 klasa → mapiramo na 6.

### Custom iPhone (tvoji skenovi)

PLY datoteke s labelama u unified 0..5 prostoru. Ručno labelijaš u CloudCompare-u. Ima RGB, NEMA intensity.

### Domain gap problem

| Izvor | Domet | Gustoća | RGB | Intensity | Šum |
|-------|-------|---------|-----|-----------|-----|
| iPhone LiDAR | ≤5m | Visoka | Da | Ne | Srednji |
| Toronto-3D | 30-50m | Srednja | Da | Da | Nizak |
| SemanticKITTI | 70m+ | Rijetka | Ne | Da | Nizak |

Modality dropout i multi-dataset trening pomažu premostiti ove razlike.

---

## Trening

### train.py

Trening loop:

1. **Podaci** — `build_datasets` iz config YAML-a → ConcatDataset za trening i validaciju
2. **Sampler** — `WeightedRandomSampler` za balansiranje kada se koristi više dataseta
3. **Optimizator** — AdamW s CosineAnnealingLR schedulerom
4. **Loss** — CrossEntropy (s class weights) + 0.5 × Lovász Softmax
5. **AMP** — Automatic Mixed Precision (float16 na GPU) za uštedi memorije
6. **Evaluacija** — per-class IoU (Intersection over Union), prosjek = mIoU

Lovász loss je posebno važan jer je boundary-aware — kažnjava pogreške na rubovima segmenata više nego u sredini.

### Config parametri (smoke test za GTX 1060 6GB)

```yaml
batch_size: 2          # manje = manje VRAM-a
crop_points: 16384     # koliko točaka po tile-u
voxel: 0.05            # veličina voxela za downsample
epochs: 15             # broj prolaza kroz podatke
steps_per_epoch: 100   # ograniči korake po epohi
lr: 0.001              # learning rate
```

---

## Interakcija i vizualizacija

### picker.py

- `pick_segment(scene, mouse_pos)` — nearest point → segment ID
- `pick_wall(scene, segment_id)` — segment → wall objekt
- `ray_to_world_point(camera, pixel)` — ray marching s KD-tree nearest upitom

### viewer.py

Open3D vizualizacija:

- Svaka klasa ima svoju boju (zelena=tlo, siva=cesta, crvena=zid, itd.)
- Sintetičke točke (dodane editiranjem) imaju drugačiji shade
- Selektirani segment je highlight-an
- Zidovi se prikazuju kao LineSet overlay (wireframe pravokutnici)

---

## Razvojne faze

### Faza 1 — Geometrija (završeno)
Predobrada, ekstrakcija tla, ravnina, klastera, značajki. Radi bez ML-a.

### Faza 2 — ML pipeline (u tijeku)
RandLA-Net model, dataset loaderi, trening loop, tiled inference.

### Faza 3 — Fuzija i zidovi (završeno u kodu)
Hibridna fuzija, wall extraction, scene objekt.

### Faza 4 — Interaktivni editor (sljedeće)
Vizualizacija, picking, operacije uređivanja, undo/redo.

### Faza 5 — Poboljšanja
- Point Transformer V3 (mIoU 80-88% vs RandLA-Net 70%)
- Više dataseta za trening
- Vlastiti iPhone skenovi s labelama

### Faza 6 — iOS deployment
Port modela na CoreML, integracija s iPhone LiDAR API-jem.

---

## Struktura projekta

```
okolis_ai/
├── configs/            # YAML konfiguracije za trening
├── datasets/           # Dataset loaderi (Toronto3D, SemanticKITTI, itd.)
│   ├── label_maps.py   # Mapiranje klasa na unified taksonomiju
│   ├── common.py       # pack_features, modality_dropout
│   ├── merged.py       # Base dataset klasa s tiling-om
│   ├── toronto3d.py    # Toronto-3D loader
│   ├── semantickitti.py # SemanticKITTI loader
│   ├── botanicgarden.py # BotanicGarden loader
│   ├── custom_iphone.py # Custom iPhone loader
│   └── builders.py     # Registry i builder iz config-a
├── editing/            # Operacije uređivanja scene
│   └── ops.py          # extend_wall, replace_terrain
├── fusion/             # Hibridna fuzija geo + ML
│   └── hybrid.py
├── geometry/           # Geometrijska analiza
│   ├── preprocess.py   # Čišćenje i priprema oblaka
│   ├── ground.py       # Ekstrakcija tla
│   ├── planes.py       # Iterativni RANSAC za ravnine
│   ├── clusters.py     # DBSCAN klasteriranje
│   └── features.py     # PCA značajke segmenata
├── interaction/        # Korisničko sučelje
│   ├── picker.py       # Odabir segmenata i zidova
│   └── viewer.py       # Open3D vizualizacija
├── io/                 # Ulaz/izlaz
│   └── ply_loader.py   # Čitanje i pisanje PLY datoteka
├── ml/                 # Machine learning
│   ├── base.py         # Segmenter ABC
│   ├── inference.py    # Tiled inference
│   └── randlanet/      # RandLA-Net implementacija
│       ├── model.py    # PyTorch model
│       └── segmenter.py # Wrapper za inference
├── scene/              # Reprezentacija scene
│   └── scene.py        # Scene dataclass
├── scripts/            # CLI skripte
│   ├── run_pipeline.py # End-to-end pipeline
│   └── verify_dataset.py # Provjera dataseta
├── segments/           # Segmenti
│   └── segment.py      # Segment dataclass
├── training/           # Trening
│   ├── train.py        # Training loop
│   └── losses.py       # CE + Lovász loss
├── tests/              # Testovi
│   └── test_smoke.py   # Smoke test
└── walls/              # Ekstrakcija zidova
    └── wall_extractor.py
```

---

## Poznate slabosti i mitigacije

| Slabost | Utjecaj | Mitigacija |
|---------|---------|-----------|
| Domain gap iPhone vs javni dataseti | Model treniran na Toronto-3D ne radi savršeno na iPhone skenovima | Modality dropout, fine-tuning na vlastitim skenovima |
| RandLA-Net nije SOTA | mIoU ~70% vs PTv3 ~85% | Upgrade na Point Transformer V3 u budućnosti |
| GTX 1060 6GB ograničenje | Mali batch, mali crop → sporiji trening | Smoke test config, cloud GPU za duži trening |
| Nema automatskog labeliranja iPhone skenova | Ručno u CloudCompare | Budući active learning pipeline |
| Ravni tereni only (grid ground) | Strme padine mogu zbuniti | Grid-based metoda s median smoothing |