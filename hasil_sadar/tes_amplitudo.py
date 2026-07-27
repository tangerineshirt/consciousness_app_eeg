from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


# ============================================================
# KONFIGURASI
# ============================================================

# Masukkan seluruh file CSV dari empat subjek di sini.
FILE_PATHS = [
    Path(
        "/home/razzannr/Documents/GitHub/consciousness_app_eeg/"
        "hasil_sadar/muse_af7_anas_20260508_102840.csv"
    ),
    Path(
        "/home/razzannr/Documents/GitHub/consciousness_app_eeg/"
        "hasil_sadar/muse_af7_sample_test_gui_02.csv"
    ),
    Path(
        "/home/razzannr/Documents/GitHub/consciousness_app_eeg/"
        "hasil_sadar/muse_af7_zurin_20260508_104304.csv"
    ),
    Path(
        "/home/razzannr/Documents/GitHub/consciousness_app_eeg/"
        "hasil_sadar/razzan_sadar.csv"
    ),
]

OUTPUT_PATH = Path(
    "/home/razzannr/Documents/GitHub/consciousness_app_eeg/"
    "hasil_sadar/hasil_pengujian_amplitudo_af7.txt"
)

COLUMN_NAME = "AF7"

# Sampling rate Muse 2
SAMPLING_RATE = 256.0

# Window 5 detik
WINDOW_DURATION = 5.0

# Filter keseluruhan rentang EEG penelitian
LOWCUT = 0.5
HIGHCUT = 45.0
FILTER_ORDER = 4


# Rentang yang disebut Naydenov et al. (2022)
PHYSIOLOGICAL_MIN = 15.0
AVERAGE_MIN = 30.0
AVERAGE_MAX = 80.0
PHYSIOLOGICAL_MAX = 150.0


# ============================================================
# MEMBACA DATA AF7
# ============================================================

def read_af7_column(csv_path: Path) -> np.ndarray:
    """
    Membaca kolom AF7 dari CSV.

    Nilai AF7 diasumsikan:
    - berasal dari Muse 2;
    - sudah dalam satuan mikrovolt;
    - belum dinormalisasi dengan z-score.
    """
    dataframe = pd.read_csv(csv_path)

    if COLUMN_NAME not in dataframe.columns:
        available_columns = ", ".join(
            map(str, dataframe.columns)
        )

        raise ValueError(
            f"Kolom '{COLUMN_NAME}' tidak ditemukan pada:\n"
            f"{csv_path}\n"
            f"Kolom yang tersedia: {available_columns}"
        )

    af7 = pd.to_numeric(
        dataframe[COLUMN_NAME],
        errors="coerce",
    )

    af7 = af7.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if af7.empty:
        raise ValueError(
            f"Kolom AF7 pada file berikut tidak memiliki "
            f"data numerik valid:\n{csv_path}"
        )

    return af7.to_numpy(dtype=float)


# ============================================================
# FILTER 0,5–45 HZ
# ============================================================

def bandpass_filter(
    signal: np.ndarray,
    sampling_rate: float,
    lowcut: float,
    highcut: float,
    order: int,
) -> np.ndarray:
    """
    Butterworth band-pass filter dengan zero-phase filtering.
    """
    nyquist_frequency = sampling_rate / 2.0

    if not 0 < lowcut < highcut < nyquist_frequency:
        raise ValueError(
            f"Rentang filter {lowcut}-{highcut} Hz tidak valid "
            f"untuk sampling rate {sampling_rate} Hz."
        )

    sos = butter(
        N=order,
        Wn=[lowcut, highcut],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )

    return sosfiltfilt(sos, signal)


# ============================================================
# MEMBAGI SINYAL MENJADI WINDOW
# ============================================================

def split_into_windows(
    signal: np.ndarray,
    sampling_rate: float,
    window_duration: float,
) -> list[np.ndarray]:
    """
    Membagi sinyal menjadi window berukuran tetap.

    Window terakhir yang tidak penuh tidak digunakan.
    """
    samples_per_window = int(
        round(sampling_rate * window_duration)
    )

    if samples_per_window <= 0:
        raise ValueError(
            "Jumlah sampel per window harus lebih besar dari nol."
        )

    total_complete_windows = (
        len(signal) // samples_per_window
    )

    if total_complete_windows == 0:
        raise ValueError(
            "Data terlalu pendek untuk menghasilkan satu window penuh."
        )

    windows = []

    for window_index in range(total_complete_windows):
        start_index = (
            window_index * samples_per_window
        )

        end_index = (
            start_index + samples_per_window
        )

        windows.append(
            signal[start_index:end_index]
        )

    return windows


# ============================================================
# AMPLITUDO PEAK-TO-PEAK
# ============================================================

def calculate_peak_to_peak(
    signal: np.ndarray,
) -> float:
    """
    Menghitung amplitudo peak-to-peak:

    A_pp = maksimum - minimum
    """
    return float(
        np.max(signal) - np.min(signal)
    )


# ============================================================
# KATEGORI AMPLITUDO
# ============================================================

def classify_amplitude(
    amplitude_uv: float,
) -> str:
    """
    Mengelompokkan amplitudo berdasarkan nilai dari
    Naydenov et al. (2022).
    """
    if amplitude_uv < PHYSIOLOGICAL_MIN:
        return "DI BAWAH 15 µV"

    if amplitude_uv < AVERAGE_MIN:
        return "15-<30 µV"

    if amplitude_uv <= AVERAGE_MAX:
        return "30-80 µV"

    if amplitude_uv <= PHYSIOLOGICAL_MAX:
        return ">80-150 µV"

    return "DI ATAS 150 µV"


# ============================================================
# ANALISIS SATU SUBJEK
# ============================================================

def analyze_subject(
    csv_path: Path,
) -> dict:
    """
    Menganalisis seluruh window dari satu file subjek.

    Tidak ada seleksi atau pembuangan window berdasarkan amplitudo.
    """
    raw_af7 = read_af7_column(
        csv_path
    )

    # Menghilangkan offset DC.
    # Satuan tetap mikrovolt.
    centered_af7 = (
        raw_af7 - np.mean(raw_af7)
    )

    filtered_af7 = bandpass_filter(
        signal=centered_af7,
        sampling_rate=SAMPLING_RATE,
        lowcut=LOWCUT,
        highcut=HIGHCUT,
        order=FILTER_ORDER,
    )

    windows = split_into_windows(
        signal=filtered_af7,
        sampling_rate=SAMPLING_RATE,
        window_duration=WINDOW_DURATION,
    )

    amplitudes = np.asarray(
        [
            calculate_peak_to_peak(window)
            for window in windows
        ],
        dtype=float,
    )

    categories = [
        classify_amplitude(amplitude)
        for amplitude in amplitudes
    ]

    category_counts = {
        "DI BAWAH 15 µV": categories.count(
            "DI BAWAH 15 µV"
        ),
        "15-<30 µV": categories.count(
            "15-<30 µV"
        ),
        "30-80 µV": categories.count(
            "30-80 µV"
        ),
        ">80-150 µV": categories.count(
            ">80-150 µV"
        ),
        "DI ATAS 150 µV": categories.count(
            "DI ATAS 150 µV"
        ),
    }

    total_windows = len(amplitudes)

    physiological_count = int(
        np.sum(
            (amplitudes >= PHYSIOLOGICAL_MIN)
            & (amplitudes <= PHYSIOLOGICAL_MAX)
        )
    )

    average_range_count = int(
        np.sum(
            (amplitudes >= AVERAGE_MIN)
            & (amplitudes <= AVERAGE_MAX)
        )
    )

    return {
        "file_path": csv_path,
        "sample_count": len(raw_af7),
        "duration_seconds": (
            len(raw_af7) / SAMPLING_RATE
        ),
        "total_windows": total_windows,
        "amplitudes": amplitudes,
        "minimum_uv": float(
            np.min(amplitudes)
        ),
        "maximum_uv": float(
            np.max(amplitudes)
        ),
        "mean_uv": float(
            np.mean(amplitudes)
        ),
        "median_uv": float(
            np.median(amplitudes)
        ),
        "standard_deviation_uv": float(
            np.std(amplitudes, ddof=0)
        ),
        "percentile_25_uv": float(
            np.percentile(amplitudes, 25)
        ),
        "percentile_75_uv": float(
            np.percentile(amplitudes, 75)
        ),
        "iqr_uv": float(
            np.percentile(amplitudes, 75)
            - np.percentile(amplitudes, 25)
        ),
        "physiological_count": physiological_count,
        "physiological_percentage": (
            physiological_count
            / total_windows
            * 100.0
        ),
        "average_range_count": average_range_count,
        "average_range_percentage": (
            average_range_count
            / total_windows
            * 100.0
        ),
        "category_counts": category_counts,
    }


# ============================================================
# FORMAT PERSENTASE KATEGORI
# ============================================================

def calculate_percentage(
    count: int,
    total: int,
) -> float:
    if total == 0:
        return 0.0

    return count / total * 100.0


# ============================================================
# MEMBUAT LAPORAN TXT
# ============================================================

def write_report(
    output_path: Path,
    subject_results: list[dict],
) -> None:
    """
    Menulis hasil per subjek dan hasil gabungan ke TXT.
    """
    all_amplitudes = np.concatenate(
        [
            result["amplitudes"]
            for result in subject_results
        ]
    )

    total_all_windows = len(
        all_amplitudes
    )

    combined_category_counts = {
        "DI BAWAH 15 µV": int(
            np.sum(
                all_amplitudes < PHYSIOLOGICAL_MIN
            )
        ),
        "15-<30 µV": int(
            np.sum(
                (all_amplitudes >= PHYSIOLOGICAL_MIN)
                & (all_amplitudes < AVERAGE_MIN)
            )
        ),
        "30-80 µV": int(
            np.sum(
                (all_amplitudes >= AVERAGE_MIN)
                & (all_amplitudes <= AVERAGE_MAX)
            )
        ),
        ">80-150 µV": int(
            np.sum(
                (all_amplitudes > AVERAGE_MAX)
                & (all_amplitudes <= PHYSIOLOGICAL_MAX)
            )
        ),
        "DI ATAS 150 µV": int(
            np.sum(
                all_amplitudes > PHYSIOLOGICAL_MAX
            )
        ),
    }

    combined_physiological_count = int(
        np.sum(
            (all_amplitudes >= PHYSIOLOGICAL_MIN)
            & (all_amplitudes <= PHYSIOLOGICAL_MAX)
        )
    )

    combined_average_count = int(
        np.sum(
            (all_amplitudes >= AVERAGE_MIN)
            & (all_amplitudes <= AVERAGE_MAX)
        )
    )

    lines = [
        "PENGUJIAN AMPLITUDO EEG MUSE 2 KANAL AF7",
        "ANALISIS WINDOW TANPA PEMISAHAN PITA",
        "=" * 88,
        "",
        "METODE",
        "-" * 88,
        f"Kolom sinyal                  : {COLUMN_NAME}",
        f"Frekuensi sampling            : {SAMPLING_RATE:.3f} Hz",
        f"Filter                        : {LOWCUT:.1f}-{HIGHCUT:.1f} Hz",
        f"Orde filter                   : {FILTER_ORDER}",
        f"Durasi window                 : {WINDOW_DURATION:.1f} detik",
        (
            "Jumlah sampel per window      : "
            f"{int(SAMPLING_RATE * WINDOW_DURATION)}"
        ),
        "Metode amplitudo               : Peak-to-peak",
        "Pemisahan pita EEG             : Tidak",
        "Seleksi/pembuangan window      : Tidak",
        "",
        "ACUAN INTERPRETASI",
        "-" * 88,
        "Di bawah 15 µV                 : Di bawah rentang fisiologis umum",
        "15 sampai <30 µV               : Amplitudo rendah",
        "30 sampai 80 µV                : Rentang rata-rata",
        ">80 sampai 150 µV              : Amplitudo tinggi dalam rentang umum",
        "Di atas 150 µV                 : Di atas rentang umum",
        "",
        "RINGKASAN PER SUBJEK",
        "=" * 88,
    ]

    for subject_index, result in enumerate(
        subject_results,
        start=1,
    ):
        counts = result["category_counts"]
        total = result["total_windows"]

        lines.extend(
            [
                "",
                f"SUBJEK {subject_index}",
                "-" * 88,
                f"File                         : {result['file_path']}",
                f"Jumlah sampel                : {result['sample_count']}",
                (
                    "Durasi rekaman               : "
                    f"{result['duration_seconds']:.3f} detik"
                ),
                f"Jumlah window                : {total}",
                "",
                "STATISTIK AMPLITUDO WINDOW",
                (
                    "Minimum                      : "
                    f"{result['minimum_uv']:.6f} µV"
                ),
                (
                    "Maksimum                     : "
                    f"{result['maximum_uv']:.6f} µV"
                ),
                (
                    "Mean                         : "
                    f"{result['mean_uv']:.6f} µV"
                ),
                (
                    "Median                       : "
                    f"{result['median_uv']:.6f} µV"
                ),
                (
                    "Standar deviasi              : "
                    f"{result['standard_deviation_uv']:.6f} µV"
                ),
                (
                    "Persentil 25                 : "
                    f"{result['percentile_25_uv']:.6f} µV"
                ),
                (
                    "Persentil 75                 : "
                    f"{result['percentile_75_uv']:.6f} µV"
                ),
                (
                    "Interquartile range          : "
                    f"{result['iqr_uv']:.6f} µV"
                ),
                "",
                "DISTRIBUSI KATEGORI",
                (
                    "Di bawah 15 µV               : "
                    f"{counts['DI BAWAH 15 µV']} "
                    f"({calculate_percentage(counts['DI BAWAH 15 µV'], total):.2f}%)"
                ),
                (
                    "15 sampai <30 µV             : "
                    f"{counts['15-<30 µV']} "
                    f"({calculate_percentage(counts['15-<30 µV'], total):.2f}%)"
                ),
                (
                    "30 sampai 80 µV              : "
                    f"{counts['30-80 µV']} "
                    f"({calculate_percentage(counts['30-80 µV'], total):.2f}%)"
                ),
                (
                    ">80 sampai 150 µV            : "
                    f"{counts['>80-150 µV']} "
                    f"({calculate_percentage(counts['>80-150 µV'], total):.2f}%)"
                ),
                (
                    "Di atas 150 µV               : "
                    f"{counts['DI ATAS 150 µV']} "
                    f"({calculate_percentage(counts['DI ATAS 150 µV'], total):.2f}%)"
                ),
                "",
                (
                    "Window dalam 15-150 µV       : "
                    f"{result['physiological_count']} dari {total} "
                    f"({result['physiological_percentage']:.2f}%)"
                ),
                (
                    "Window dalam 30-80 µV        : "
                    f"{result['average_range_count']} dari {total} "
                    f"({result['average_range_percentage']:.2f}%)"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "HASIL GABUNGAN SELURUH SUBJEK",
            "=" * 88,
            (
                "Jumlah subjek                 : "
                f"{len(subject_results)}"
            ),
            (
                "Jumlah seluruh window         : "
                f"{total_all_windows}"
            ),
            "",
            "STATISTIK GABUNGAN",
            (
                "Minimum                       : "
                f"{np.min(all_amplitudes):.6f} µV"
            ),
            (
                "Maksimum                      : "
                f"{np.max(all_amplitudes):.6f} µV"
            ),
            (
                "Mean                          : "
                f"{np.mean(all_amplitudes):.6f} µV"
            ),
            (
                "Median                        : "
                f"{np.median(all_amplitudes):.6f} µV"
            ),
            (
                "Standar deviasi               : "
                f"{np.std(all_amplitudes):.6f} µV"
            ),
            (
                "Persentil 25                  : "
                f"{np.percentile(all_amplitudes, 25):.6f} µV"
            ),
            (
                "Persentil 75                  : "
                f"{np.percentile(all_amplitudes, 75):.6f} µV"
            ),
            (
                "Interquartile range           : "
                f"{np.percentile(all_amplitudes, 75) - np.percentile(all_amplitudes, 25):.6f} µV"
            ),
            "",
            "DISTRIBUSI GABUNGAN",
        ]
    )

    for category_name, count in combined_category_counts.items():
        percentage = calculate_percentage(
            count,
            total_all_windows,
        )

        lines.append(
            f"{category_name:<30}: "
            f"{count} window ({percentage:.2f}%)"
        )

    lines.extend(
        [
            "",
            (
                "Total window dalam 15-150 µV  : "
                f"{combined_physiological_count} "
                f"dari {total_all_windows} "
                f"({calculate_percentage(combined_physiological_count, total_all_windows):.2f}%)"
            ),
            (
                "Total window dalam 30-80 µV   : "
                f"{combined_average_count} "
                f"dari {total_all_windows} "
                f"({calculate_percentage(combined_average_count, total_all_windows):.2f}%)"
            ),
            "",
            "CATATAN INTERPRETASI",
            "=" * 88,
            (
                "1. Tidak ada window yang dihapus atau dipilih berdasarkan "
                "nilai amplitudonya."
            ),
            (
                "2. Rentang 15-150 µV merupakan rentang amplitudo EEG "
                "fisiologis umum yang disebut Naydenov et al. (2022)."
            ),
            (
                "3. Rentang 30-80 µV dinyatakan sebagai batas rata-rata, "
                "bukan batas mutlak valid atau tidak valid."
            ),
            (
                "4. Paper tidak menjelaskan secara eksplisit apakah nilai "
                "amplitudo dihitung dengan RMS, peak, atau peak-to-peak."
            ),
            (
                "5. Peak-to-peak digunakan sebagai definisi operasional "
                "dalam pengujian ini."
            ),
            (
                "6. Hasil ini merupakan pemeriksaan kewajaran amplitudo "
                "fisiologis awal dan bukan validasi klinis Muse 2."
            ),
            "",
            "REFERENSI",
            "=" * 88,
            (
                "Naydenov, C., Yordanova, A., & Mancheva, V. (2022). "
                "Methodology for EEG and Reference Values of the Software "
                "Analysis. Open Access Macedonian Journal of Medical "
                "Sciences, 10(B), 2351-2354."
            ),
            "DOI: 10.3889/oamjms.2022.10751",
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# PROGRAM UTAMA
# ============================================================

def main() -> None:
    for file_path in FILE_PATHS:
        if not file_path.exists():
            raise FileNotFoundError(
                f"File tidak ditemukan:\n{file_path}"
            )

    subject_results = []

    for subject_index, file_path in enumerate(
        FILE_PATHS,
        start=1,
    ):
        print(
            f"Menganalisis subjek {subject_index}: "
            f"{file_path.name}"
        )

        result = analyze_subject(
            file_path
        )

        subject_results.append(
            result
        )

    write_report(
        output_path=OUTPUT_PATH,
        subject_results=subject_results,
    )

    print("=" * 70)
    print("Analisis selesai.")
    print(f"Jumlah subjek : {len(subject_results)}")
    print(f"File hasil    : {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()