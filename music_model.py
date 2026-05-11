"""
CodeAlpha Task 3: Music Generation with AI
═══════════════════════════════════════════
Advanced LSTM + Transformer Hybrid Music Generation Engine

Architecture:
  • Bidirectional LSTM with stacked layers
  • Multi-Head Self-Attention for long-range dependencies
  • Nucleus (top-p) sampling with temperature control
  • Genre-conditioned generation
  • Expressive MIDI export with velocity curves & duration mapping
  • Harmonic analysis & chord progression awareness
  • Real-time training with EarlyStopping
"""

import numpy as np
import os
import random
import warnings
import math
import logging

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense, Dropout, Embedding,
    LayerNormalization, MultiHeadAttention, Input,
    GlobalAveragePooling1D, Add, Flatten, Reshape,
    Attention
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import mido
from mido import MidiFile, MidiTrack, Message

logger = logging.getLogger(__name__)

# ─── Expanded Musical Training Corpus ────────────────────────────────────────

GENRE_DATA = {
    "classical": {
        "sequences": [
            # Bach-style counterpoint & inventions
            [60,62,64,65,67,69,71,72,71,69,67,65,64,62,60,59,60],
            [72,71,69,67,65,64,62,60,62,64,65,67,69,71,72,74,72],
            [60,64,67,72,67,64,60,62,65,69,72,69,65,62,60,64,67],
            [67,67,67,63,65,65,65,60,64,64,64,60,62,62,62,59],
            # Beethoven motifs
            [67,67,67,63,0,65,65,65,60,0,64,64,64,61,0,60],
            [60,62,64,65,64,62,60,67,65,64,62,64,65,67,65,64],
            [72,71,69,67,65,64,62,60,62,64,65,67,69,71,72,74],
            # Mozart sonata fragments
            [60,62,64,65,67,65,64,62,60,62,64,65,67,69,71,72,71,69],
            [64,67,71,72,71,67,64,65,69,72,71,69,65,64,60,62,64,65],
            [60,64,67,60,64,67,60,65,69,60,65,69,60,64,67,72,74,72],
            # Chopin chromatic / chromatic voice leading
            [60,61,62,63,64,65,66,67,68,69,70,71,72,71,70,69,68,67],
            [72,71,70,69,68,67,66,65,64,63,62,61,60,62,64,65,67,69],
            # Arpeggios I-IV-V-I
            [60,64,67,72,67,64,60,65,69,72,69,65,62,65,69,74,69,65],
            [48,60,64,67,60,64,67,60,50,62,65,69,62,65,69,62],
            # Alberti bass pattern (extended)
            [60,67,64,67,60,67,64,67,65,69,65,69,62,65,62,65,60,67],
            [48,55,52,55,48,55,52,55,50,57,53,57,50,57,53,57],
        ],
        "description": "Inspired by Bach, Mozart, Beethoven and Chopin — rich harmonic voice leading and melodic counterpoint.",
        "tempo_range": [80, 140],
        "scale": "major/minor",
        "mood": "Structured & Emotive",
        "key_notes": [60, 62, 64, 65, 67, 69, 71, 72],  # C major
        "chord_roots": [60, 65, 67, 62],  # I IV V ii
    },
    "jazz": {
        "sequences": [
            # ii-V-I progressions, chromatic runs
            [60,63,67,70,72,70,67,63,60,62,65,68,70,72,70,68],
            [62,65,69,72,74,72,69,65,62,60,63,67,70,72,70,67],
            [67,70,74,77,79,77,74,70,67,65,68,72,75,77,75,72],
            # Blues scale runs
            [60,63,65,66,67,70,72,70,67,66,65,63,60,63,65,67],
            [72,70,69,67,65,63,60,63,65,67,69,70,72,75,72,70],
            # Bebop lines
            [60,62,64,65,67,65,64,62,60,63,65,67,70,72,70,67,65,63],
            [67,65,64,62,60,62,64,65,67,70,72,70,67,65,63,62,60],
            [60,61,62,63,64,65,66,67,65,63,62,60,63,65,67,70,72],
            # Swing triplet feel encoded as eighth notes
            [60,63,67,63,60,65,68,65,62,65,69,65,60,64,67,64],
            [67,70,74,70,67,69,72,69,67,65,68,65,60,63,67,63],
            # Extended voicings (9ths, 11ths)
            [60,64,67,70,74,72,70,67,64,62,65,69,72,76,74,72],
            [62,65,69,72,76,74,72,69,65,60,63,67,70,74,72,70],
        ],
        "description": "Blues scales, ii-V-I progressions and bebop chromatic lines inspired by Miles Davis and Charlie Parker.",
        "tempo_range": [100, 180],
        "scale": "blues/mixolydian",
        "mood": "Improvisational & Swinging",
        "key_notes": [60, 63, 65, 67, 70, 72],  # blues scale C
        "chord_roots": [60, 65, 67],
    },
    "ambient": {
        "sequences": [
            # Pentatonic, slow movement
            [60,64,67,72,76,79,76,72,67,64,60,62,65,69,72],
            [57,60,64,67,72,67,64,60,57,55,59,62,67,72,67,62],
            [62,65,69,74,77,74,69,65,62,60,64,67,72,76,72,67],
            [64,67,71,76,79,76,71,67,64,62,65,69,74,77,74,69],
            # Sustained drone-like patterns
            [60,0,0,64,0,0,67,0,0,72,0,0,67,0,0,64,0,0],
            [57,0,0,0,62,0,0,0,65,0,0,0,69,0,0,0,72,0],
            # Whole-tone dreamy movement
            [60,62,64,66,68,70,72,70,68,66,64,62,60,62,64,66],
            [67,69,71,73,75,77,79,77,75,73,71,69,67,69,71,73],
            # Lydian mode (ethereal)
            [60,62,64,66,67,69,71,72,71,69,67,66,64,62,60,62],
            [65,67,69,71,72,74,76,77,76,74,72,71,69,67,65,67],
        ],
        "description": "Pentatonic and whole-tone progressions with sustained resonance — inspired by Brian Eno and Harold Budd.",
        "tempo_range": [50, 90],
        "scale": "pentatonic/whole-tone",
        "mood": "Ethereal & Meditative",
        "key_notes": [60, 62, 64, 67, 69],  # pentatonic
        "chord_roots": [60, 65, 62],
    },
    "blues": {
        "sequences": [
            # 12-bar blues patterns
            [60,63,65,66,67,63,60,65,67,65,63,60],
            [65,68,70,71,72,68,65,70,72,70,68,65],
            [67,70,72,73,74,70,67,72,74,72,70,67],
            [60,63,65,66,67,65,63,60,67,65,63,60,65,67,65,63],
            [72,70,68,67,65,63,60,63,65,67,68,70,72,70,68,67],
            # Slide guitar emulation
            [60,61,62,63,65,63,62,61,60,62,63,65,66,65,63,62],
            [67,68,69,70,72,70,69,68,67,69,70,72,73,72,70,69],
            # Shuffle feel
            [60,63,60,65,63,60,67,65,60,63,65,67,65,63,60],
            # Call and response
            [60,63,65,67,0,0,65,63,60,65,67,70,0,0,67,65],
            [67,70,72,75,0,0,72,70,67,65,67,70,72,70,67,65],
        ],
        "description": "Authentic 12-bar blues progressions and slide guitar licks inspired by Robert Johnson and BB King.",
        "tempo_range": [70, 130],
        "scale": "blues pentatonic",
        "mood": "Soulful & Expressive",
        "key_notes": [60, 63, 65, 66, 67, 70],
        "chord_roots": [60, 65, 67],
    },
    "romantic": {
        "sequences": [
            # Liszt / Rachmaninoff inspired
            [60,64,67,72,76,79,76,72,67,64,60,55,59,62,67,71],
            [64,68,71,75,80,75,71,68,64,62,65,69,74,77,74,69],
            [60,63,67,72,75,72,67,63,60,58,62,65,70,75,70,65],
            # Sweeping arpeggios
            [48,52,55,60,64,67,72,76,79,76,72,67,64,60,55,52],
            [50,53,57,62,65,69,74,77,81,77,74,69,65,62,57,53],
            # Lyrical melody
            [72,74,76,77,79,77,76,74,72,71,69,67,69,71,72,74,76,77],
            [67,69,71,72,74,72,71,69,67,65,64,62,64,65,67,69,71,72],
            # Dramatic chromatic descent
            [72,71,70,69,68,67,66,65,64,63,62,61,60,62,64,65,67,69],
            # Nocturne-style
            [64,67,72,76,79,76,72,67,64,67,71,74,76,74,71,67],
            [60,64,69,72,76,79,81,79,76,72,69,64,60,64,67,71],
        ],
        "description": "Grand sweeping arpeggios and lyrical melodies from the Romantic tradition — Liszt, Chopin and Rachmaninoff.",
        "tempo_range": [60, 110],
        "scale": "chromatic/minor",
        "mood": "Grand & Passionate",
        "key_notes": [60, 62, 63, 65, 67, 68, 70, 72],  # harmonic minor
        "chord_roots": [60, 65, 67, 68],
    },
}


# ─── Genre Info ──────────────────────────────────────────────────────────────

def get_genre_info() -> dict:
    """Return genre metadata for the frontend."""
    result = {}
    for genre, data in GENRE_DATA.items():
        result[genre] = {
            "description": data["description"],
            "tempo_range":  data["tempo_range"],
            "scale":        data["scale"],
            "mood":         data["mood"],
        }
    return result


# ─── Data Preprocessing ──────────────────────────────────────────────────────

def prepare_sequences(sequences: list, seq_length: int = 16):
    """
    Encode note sequences into supervised LSTM training pairs.
    Returns: X, y, note_to_int, int_to_note, unique_notes
    """
    all_notes = []
    for seq in sequences:
        all_notes.extend(seq)

    unique_notes = sorted(set(all_notes))
    note_to_int  = {n: i for i, n in enumerate(unique_notes)}
    int_to_note  = {i: n for n, i in note_to_int.items()}

    encoded = [note_to_int[n] for n in all_notes]
    X, y = [], []
    for i in range(len(encoded) - seq_length):
        X.append(encoded[i : i + seq_length])
        y.append(encoded[i + seq_length])

    X = np.array(X, dtype=np.int32)
    y = tf.keras.utils.to_categorical(y, num_classes=len(unique_notes))
    return X, y, note_to_int, int_to_note, unique_notes


# ─── Model Architecture ──────────────────────────────────────────────────────

def build_model(n_notes: int, seq_length: int = 16, complexity: int = 3) -> Model:
    """
    Build scalable music generation model.

    complexity 1-2 : Single LSTM  (fast, low memory)
    complexity 3   : Stacked LSTM (balanced, default)
    complexity 4-5 : Bidirectional LSTM + Self-Attention (research grade)
    """
    if complexity <= 2:
        # ── Lightweight ──────────────────────────────────────────────
        model = Sequential([
            Embedding(n_notes, 64, input_length=seq_length),
            LSTM(128, return_sequences=False),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(n_notes, activation="softmax"),
        ], name="LightweightLSTM")

    elif complexity == 3:
        # ── Balanced Stacked LSTM ─────────────────────────────────────
        model = Sequential([
            Embedding(n_notes, 96, input_length=seq_length),
            LSTM(256, return_sequences=True),
            LayerNormalization(),
            Dropout(0.3),
            LSTM(256),
            Dropout(0.3),
            Dense(128, activation="relu"),
            Dense(n_notes, activation="softmax"),
        ], name="StackedLSTM")

    elif complexity == 4:
        # ── Deep Bidirectional ────────────────────────────────────────
        model = Sequential([
            Embedding(n_notes, 128, input_length=seq_length),
            Bidirectional(LSTM(256, return_sequences=True)),
            LayerNormalization(),
            Dropout(0.3),
            Bidirectional(LSTM(128)),
            Dense(256, activation="relu"),
            Dropout(0.2),
            Dense(128, activation="relu"),
            Dense(n_notes, activation="softmax"),
        ], name="BiLSTM")

    else:
        # ── Transformer-enhanced LSTM (complexity 5) ──────────────────
        # Functional API for residual connections + attention
        inp = Input(shape=(seq_length,), name="note_input")
        x   = Embedding(n_notes, 128, name="embedding")(inp)

        # Self-attention block
        attn_out = MultiHeadAttention(
            num_heads=4, key_dim=32, name="self_attention"
        )(x, x)
        x = Add(name="residual_1")([x, attn_out])
        x = LayerNormalization(name="ln_1")(x)

        # Bidirectional LSTM
        x = Bidirectional(LSTM(256, return_sequences=True), name="bilstm_1")(x)
        x = LayerNormalization(name="ln_2")(x)
        x = Dropout(0.3)(x)

        x = Bidirectional(LSTM(128, return_sequences=False), name="bilstm_2")(x)
        x = Dense(256, activation="relu", name="dense_1")(x)
        x = Dropout(0.2)(x)
        x = Dense(128, activation="relu", name="dense_2")(x)
        out = Dense(n_notes, activation="softmax", name="note_output")(x)

        model = Model(inputs=inp, outputs=out, name="TransformerLSTM")

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
        metrics=["accuracy"],
    )
    return model


# ─── Training ────────────────────────────────────────────────────────────────

def train_model(genre: str = "classical", complexity: int = 3):
    """
    Train on genre-specific sequences.
    Returns: model, note_to_int, int_to_note, unique_notes, X
    """
    data = GENRE_DATA.get(genre, GENRE_DATA["classical"])
    seqs = data["sequences"]

    # Epoch schedule: scales with complexity but bounded for responsiveness
    epoch_map = {1: 25, 2: 40, 3: 70, 4: 100, 5: 140}
    epochs     = epoch_map.get(complexity, 70)
    seq_length = 16

    X, y, note_to_int, int_to_note, unique_notes = prepare_sequences(seqs, seq_length)
    n_notes = len(unique_notes)

    logger.info(
        f"Training {genre} model | notes={n_notes} | "
        f"samples={len(X)} | complexity={complexity} | epochs={epochs}"
    )

    model = build_model(n_notes, seq_length, complexity)

    callbacks = [
        EarlyStopping(
            monitor="loss", patience=12,
            restore_best_weights=True, verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="loss", factor=0.5,
            patience=6, min_lr=1e-5, verbose=0,
        ),
    ]

    batch_size = 64 if complexity >= 4 else 32
    model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    return model, note_to_int, int_to_note, unique_notes, X


# ─── Nucleus Sampling ────────────────────────────────────────────────────────

def nucleus_sample(probs: np.ndarray, p: float = 0.9) -> int:
    """
    Top-p (nucleus) sampling for natural, non-repetitive note selection.
    Sorts by probability, accumulates until threshold p, then samples.
    """
    sorted_idx   = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumulative   = np.cumsum(sorted_probs)
    cutoff       = int(np.searchsorted(cumulative, p)) + 1
    cutoff       = max(1, min(cutoff, len(probs)))

    top_idx   = sorted_idx[:cutoff]
    top_probs = probs[top_idx]
    top_probs = top_probs / top_probs.sum()   # renormalise

    return int(np.random.choice(top_idx, p=top_probs))


def generate_notes(
    model, note_to_int: dict, int_to_note: dict,
    unique_notes: list, X: np.ndarray,
    num_notes: int = 64, temperature: float = 1.0,
) -> list:
    """
    Autoregressive generation with temperature + nucleus sampling.
    Seeds from a random training pattern for coherent style continuity.
    """
    seq_length = X.shape[1]
    seed_idx   = random.randint(0, len(X) - 1)
    pattern    = list(X[seed_idx])
    generated  = []

    # Dynamic nucleus threshold: tighter at low temp, broader at high temp
    nucleus_p = float(np.clip(0.72 + temperature * 0.16, 0.5, 0.98))

    for step in range(num_notes):
        inp  = np.array([pattern[-seq_length:]], dtype=np.int32)
        pred = model.predict(inp, verbose=0)[0].astype("float64")

        # Temperature scaling in log-space (numerically stable)
        log_pred = np.log(np.maximum(pred, 1e-10)) / temperature
        log_pred -= log_pred.max()           # shift for stability
        pred      = np.exp(log_pred)
        pred      = pred / pred.sum()

        note_idx = nucleus_sample(pred, nucleus_p)
        generated.append(int_to_note[note_idx])
        pattern.append(note_idx)

    return generated


# ─── Expressive MIDI Construction ────────────────────────────────────────────

INSTRUMENT_MAP = {
    "classical": 0,    # Acoustic Grand Piano
    "jazz":      25,   # Acoustic Guitar (Steel)
    "ambient":   88,   # Pad 1 (New Age)
    "blues":     24,   # Acoustic Guitar (Nylon)
    "romantic":  40,   # Violin
}


def build_velocity_curve(num_notes: int, genre: str) -> list:
    """
    Construct expressive velocity envelope with micro-humanisation.
    Follows stylistic dynamics per genre (e.g. ambient is softer, romantic swells).
    """
    patterns = {
        "classical": [64, 72, 80, 88, 92, 88, 80, 72],
        "jazz":      [70, 80, 65, 85, 70, 90, 65, 80],
        "ambient":   [44, 48, 52, 56, 52, 48, 44, 42],
        "blues":     [80, 70, 85, 70, 80, 65, 85, 70],
        "romantic":  [56, 68, 80, 92, 104, 96, 84, 72],
    }
    pat = patterns.get(genre, patterns["classical"])
    velocities = []
    for i in range(num_notes):
        base  = pat[i % len(pat)]
        human = random.randint(-6, 6)
        velocities.append(max(28, min(118, base + human)))
    return velocities


def build_duration_map(notes: list, genre: str, tempo_bpm: int = 120) -> list:
    """
    Assign musically appropriate note durations based on genre feel.
    Uses ticks-per-beat (480 TPB standard).
    """
    tpb     = 480
    sixteenth = tpb // 4
    eighth    = tpb // 2
    quarter   = tpb
    dotted_q  = int(tpb * 1.5)
    half      = tpb * 2

    duration_pools = {
        "ambient":   [quarter, dotted_q, half, half, dotted_q, quarter],
        "jazz":      [eighth, eighth, quarter, eighth, sixteenth, eighth],
        "blues":     [eighth, sixteenth, quarter, eighth, eighth, dotted_q],
        "romantic":  [quarter, dotted_q, half, quarter, eighth, dotted_q],
        "classical": [quarter, eighth, dotted_q, quarter, eighth, quarter],
    }
    pool = duration_pools.get(genre, duration_pools["classical"])

    durations = []
    for i, note in enumerate(notes):
        if note == 0:                   # rest
            durations.append(quarter)
        else:
            d = pool[i % len(pool)]
            # Phrase-ending longer note every 8 steps
            if (i + 1) % 8 == 0:
                d = dotted_q if genre != "ambient" else half
            durations.append(d)
    return durations


def notes_to_midi(
    notes: list,
    output_path: str,
    tempo_bpm: int = 120,
    genre: str = "classical",
) -> str:
    """
    Convert a list of MIDI note numbers to an expressive .mid file.
    Includes: instrument selection, velocity curves, duration mapping,
    tempo & track name meta-messages.
    """
    mid   = MidiFile(type=0, ticks_per_beat=480)
    track = MidiTrack()
    mid.tracks.append(track)

    # Meta messages
    tempo = mido.bpm2tempo(tempo_bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    track.append(
        mido.MetaMessage(
            "track_name",
            name=f"AI {genre.title()} Composition - CodeAlpha",
            time=0,
        )
    )

    # Instrument
    instrument = INSTRUMENT_MAP.get(genre, 0)
    track.append(Message("program_change", channel=0, program=instrument, time=0))

    velocities = build_velocity_curve(len(notes), genre)
    durations  = build_duration_map(notes, genre, tempo_bpm)

    for i, note in enumerate(notes):
        vel = velocities[i]
        dur = durations[i]

        if note == 0:
            # Rest encoded as zero-velocity note-on + note-off
            track.append(Message("note_on",  note=60, velocity=0,   time=0))
            track.append(Message("note_off", note=60, velocity=0,   time=dur))
        else:
            note = max(0, min(127, int(note)))   # clamp to MIDI range
            track.append(Message("note_on",  note=note, velocity=vel, time=0))
            track.append(Message("note_off", note=note, velocity=0,   time=dur))

    mid.save(output_path)
    logger.info(f"MIDI saved → {output_path}  ({len(notes)} notes, {genre}, {tempo_bpm} BPM)")
    return output_path


# ─── Harmonic Analysis ────────────────────────────────────────────────────────

NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def analyze_notes(notes: list) -> dict:
    """
    Lightweight harmonic analysis of generated note list.
    Returns note distribution, most common notes, estimated key.
    """
    valid = [n for n in notes if n > 0]
    if not valid:
        return {"key": "unknown", "note_distribution": {}, "pitch_classes": {}}

    pitch_classes = {}
    for n in valid:
        pc = NOTE_NAMES[n % 12]
        pitch_classes[pc] = pitch_classes.get(pc, 0) + 1

    sorted_pc = sorted(pitch_classes.items(), key=lambda x: -x[1])
    most_common = sorted_pc[0][0] if sorted_pc else "C"

    # Naïve key detection: highest-frequency pitch class as tonic
    estimated_key = most_common

    # Octave distribution
    octave_dist = {}
    for n in valid:
        oct_num = (n // 12) - 1
        octave_dist[str(oct_num)] = octave_dist.get(str(oct_num), 0) + 1

    return {
        "key":              estimated_key,
        "pitch_classes":    dict(sorted_pc[:8]),
        "octave_range":     [min(valid), max(valid)],
        "note_count":       len(valid),
        "rest_count":       notes.count(0),
        "unique_pitches":   len(set(valid)),
        "octave_dist":      octave_dist,
    }


def notes_to_labels(notes: list) -> list:
    """Convert MIDI note numbers to human-readable labels like 'C4', 'G#5'."""
    labels = []
    for n in notes:
        if n == 0:
            labels.append("REST")
        else:
            name = NOTE_NAMES[n % 12]
            octave = (n // 12) - 1
            labels.append(f"{name}{octave}")
    return labels


# ─── Main Pipeline ────────────────────────────────────────────────────────────

def generate_music(
    genre: str = "classical",
    num_notes: int = 64,
    temperature: float = 0.8,
    tempo: int = 120,
    output_path: str = "output.mid",
    complexity: int = 3,
) -> tuple:
    """
    Full generation pipeline:
      1. Train LSTM model on genre data
      2. Generate note sequence via nucleus sampling
      3. Export expressive MIDI file

    Returns:
        (notes: list[int], output_path: str)
    """
    model, note_to_int, int_to_note, unique_notes, X = train_model(genre, complexity)

    notes = generate_notes(
        model, note_to_int, int_to_note, unique_notes, X,
        num_notes=num_notes,
        temperature=temperature,
    )

    notes_to_midi(notes, output_path, tempo_bpm=tempo, genre=genre)
    return notes, output_path


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    notes, path = generate_music(
        genre="classical", num_notes=32, output_path="/tmp/test_codealpha.mid"
    )
    print(f"✅ Generated {len(notes)} notes → {path}")
    print(f"   Notes[:10]: {notes[:10]}")
    analysis = analyze_notes(notes)
    print(f"   Key: {analysis['key']}  Unique pitches: {analysis['unique_pitches']}")
