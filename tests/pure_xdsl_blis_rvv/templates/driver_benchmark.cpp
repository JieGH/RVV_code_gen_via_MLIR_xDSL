#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <tuple>
#include <cstdint>

#include "generated_kernels.h"

#if defined(DTYPE_INT8)
    typedef int8_t A_TYPE;
    typedef int8_t B_TYPE;
    typedef int32_t C_TYPE;
    #define A_CANARY ((A_TYPE)-123)
    #define B_CANARY ((B_TYPE)-123)
    #define C_CANARY ((C_TYPE)-123456)
    #define GET_ERR_TOL(k) 0.0
#else
    typedef float A_TYPE;
    typedef float B_TYPE;
    typedef float C_TYPE;
    #define A_CANARY -123456.0f
    #define B_CANARY -123456.0f
    #define C_CANARY -123456.0f
    #define GET_ERR_TOL(k) (std::max(1e-5, (double)(k) * 1e-6))
#endif

extern "C" {
#include "../../modelLevel/model_level.h"
}

using namespace std;

static const char* PERF_REGIONS[] = {"Vector_Pack", "Scalar_Pack", "Compute"};

// Issue #42: one raw event asked for with --raw-events.
struct RawEvent {
    string name;
    uint64_t code;
};

static string hex_code(uint64_t code) {
    ostringstream s;
    s << "0x" << std::hex << code;
    return s.str();
}

// Parses --raw-events: name=code items separated by commas, where an item may also be the
// preset name. Returns "" or what is wrong with the list.
static string parse_raw_events(const string& arg, vector<RawEvent>& out) {
    const string form = "--raw-events takes name=code items separated by commas, or the preset " +
                        string(PERF_X60_L2_PRESET_NAME);
    size_t start = 0;
    for (;;) {
        size_t comma = arg.find(',', start);
        string item = arg.substr(start, comma == string::npos ? string::npos : comma - start);
        if (item == PERF_X60_L2_PRESET_NAME) {
            for (int i = 0; i < PERF_X60_L2_PRESET_COUNT; ++i)
                out.push_back({PERF_X60_L2_PRESET[i].name, PERF_X60_L2_PRESET[i].code});
        } else {
            size_t eq = item.find('=');
            if (eq == string::npos || eq == 0 || eq + 1 == item.size())
                return form + " (got '" + item + "').";
            string name = item.substr(0, eq), code_s = item.substr(eq + 1);
            for (char ch : name)
                if (!(isalnum((unsigned char) ch) || ch == '_'))
                    return "--raw-events: the event name '" + name + "' may only hold letters, digits and _.";
            char* end = nullptr;
            uint64_t code = std::strtoull(code_s.c_str(), &end, 0);
            if (end == code_s.c_str() || *end != '\0' || code_s[0] == '-')
                return "--raw-events: '" + code_s + "' is not an event code (a number such as 0xb9 or 185).";
            out.push_back({name, code});
        }
        if (comma == string::npos) break;
        start = comma + 1;
    }
    for (size_t i = 0; i < out.size(); ++i)
        for (size_t j = 0; j < i; ++j)
            if (out[i].name == out[j].name) return "--raw-events: '" + out[i].name + "' is named twice.";
    if ((int) out.size() > PERF_MAX_RAW)
        return "--raw-events: at most " + to_string(PERF_MAX_RAW) + " events fit in the raw group, got " +
               to_string(out.size()) + ".";
    return "";
}

// Issue #14: an uninstrumented build has no inner timers and no counters, so its
// breakdown columns are nan. Its logs get their own names so a noinstr run and an
// instrumented run of the same project never append to the same file.
#if RVV_INSTRUMENTED
#define RVV_LOG_SUFFIX ""
#else
#define RVV_LOG_SUFFIX "_noinstr"
#endif

static void write_perf_metrics(ofstream& out, const perf_metrics& m, bool valid) {
    double nan_v = std::nan("");
    double vals[] = {m.cycles, m.instructions, m.ipc, m.l1d_read_miss_pct, m.l1d_write_miss_pct, m.dtlb_misses,
                     m.stall_be_pct, m.stall_fe_pct, m.branch_misses};
    for (double v : vals) out << "," << (valid ? v : nan_v);
}

static double cv_pct_of(vector<double> vals) {
    if (vals.empty()) return std::nan("");
    double mean = 0.0, sq = 0.0;
    for (double v : vals) mean += v;
    mean /= vals.size();
    for (double v : vals) sq += (v - mean) * (v - mean);
    sort(vals.begin(), vals.end());
    double median = vals[vals.size() / 2];
    return median > 0 ? 100.0 * std::sqrt(sq / vals.size()) / median : std::nan("");
}

// Terminal-report helpers (issue #9 introduced the live per-variant block; it has since been
// reworded so every figure says what it measures, with a [bracketed note] explaining what is
// inside instead of a bare abbreviation). The offline comparison.md built by summarize_run.py
// keeps its own, more compact column names.
static string short_variant_name(const string& name) {
    // "MR16_NR15" -> "16x15"
    string s = name;
    size_t pos = s.find("MR");
    if (pos != string::npos) s.replace(pos, 2, "");
    pos = s.find("_NR");
    if (pos != string::npos) s.replace(pos, 3, "x");
    return s;
}

static string num(double v, int decimals) {
    ostringstream ss;
    ss << fixed << setprecision(decimals) << v;
    return ss.str();
}

// One "  |    label : value  [note]" row of the per-variant block. `note` says what is inside
// the value; an empty label continues the row above.
static void report_row(const string& label, const string& value, const string& note = "") {
    cout << "  |    " << left << setw(18) << label << (label.empty() ? "  " : ": ") << value;
    if (!note.empty()) cout << "  [" << note << "]";
    cout << "\n";
}

// What the speed figures divide by, per build (issue #14): the whole GEMM call in the
// uninstrumented build; the inner pack + compute timers in the instrumented one.
#if RVV_INSTRUMENTED
static const char* TIMED_VALUE = "pack timer + compute timer per rep";
static const char* TIMED_NOTE = "inner timers; gaps between them are excluded; they cost 11-17%: DEBUG build, do not compare to the reference";
static const char* MEAN_TIME_EXTRA = ", outer wall clock with the timers' cost included";
#else
static const char* TIMED_VALUE = "the whole GEMM call per rep";
static const char* TIMED_NOTE = "pack A and B + microkernel compute + loop overhead; same region the reference design times";
static const char* MEAN_TIME_EXTRA = "";
#endif

static string pct_of_total(double part_ms, double compute_ms) {
    double total = part_ms + compute_ms;
    if (total <= 0) return "N/A";
    ostringstream ss;
    ss << fixed << setprecision(1) << (100.0 * part_ms / total) << "%";
    return ss.str();
}

static string gain_pct(double scalar_ms, double vector_ms) {
    if (scalar_ms <= 0) return "N/A";
    double gain = 100.0 * (scalar_ms - vector_ms) / scalar_ms;
    ostringstream ss;
    ss << (gain >= 0 ? "+" : "") << fixed << setprecision(1) << gain << "%";
    return ss.str();
}

static string perf_field(bool valid, double v, int decimals, const char* suffix) {
    if (!valid || v != v) return "N/A";  // v != v catches NaN
    ostringstream ss;
    ss << fixed << setprecision(decimals) << v << suffix;
    return ss.str();
}

// Generate random data
template <typename T>
void fill_random(T* ptr, size_t size) {
    for (size_t i = 0; i < size; ++i) {
#if defined(DTYPE_INT8)
        ptr[i] = static_cast<T>((rand() % 10) - 5);
#else
        ptr[i] = static_cast<T>(rand()) / RAND_MAX;
#endif
    }
}

// Aligned allocation
template <typename T>
T* alloc_aligned(size_t size_elems) {
    size_t bytes = size_elems * sizeof(T);
    // Size must be a multiple of alignment for aligned_alloc
    size_t alloc_bytes = (bytes + 63) & ~63;
    void* ptr = aligned_alloc(64, alloc_bytes);
    if (!ptr) {
        cerr << "Memory allocation failed" << endl;
        exit(1);
    }
    return static_cast<T*>(ptr);
}

// Zeroed like the reference design's calloc'd Ac/Bc (issue #18). The packers write only the
// valid slots of a partial panel, and the dispatch no longer zeroes the pad slots inside the
// timed region, so the pad must start out finite. Done once here, outside every timer.
template <typename T>
T* alloc_aligned_zeroed(size_t size_elems) {
    T* ptr = alloc_aligned<T>(size_elems);
    memset(ptr, 0, size_elems * sizeof(T));
    return ptr;
}

// ----------------------------------------------------------------------------
// LEGACY REFERENCE GEMM
// ----------------------------------------------------------------------------
#define Acol(a1,a2)  A[ (a2)*(ldA)+(a1) ]
#define Bcol(a1,a2)  B[ (a2)*(ldB)+(a1) ]
#define Ccol(a1,a2)  C[ (a2)*(ldC)+(a1) ]
#define Crow(a1,a2)  C[ (a1)*(ldC)+(a2) ]
#define Arow(a1,a2)  A[ (a1)*(ldA)+(a2) ]
#define Brow(a1,a2)  B[ (a1)*(ldB)+(a2) ]

void gemm_ref( char orderA, char orderB, char orderC, 
           char transA, char transB, 
           size_t m, size_t n, size_t k, 
           float alpha, A_TYPE *A, int ldA, 
                        B_TYPE *B, int ldB, 
           float beta,  C_TYPE *C, int ldC ){

   size_t    ic, jc, i, j, p;
   C_TYPE  zero = 0, one = 1, tmp;

   // Quick return if possible
  if ( (m==0)||(n==0) || (((alpha==zero) || (k==0)) && (beta==one)) )
    return;

  // Quick gemm if alpha==0.0
  if ( alpha==zero ) {
    if ( beta==zero )
      if ( orderC=='C' ) {
        for ( jc=0; jc<n; jc++ )
          for ( ic=0; ic<m; ic++ )
            Ccol(ic,jc) = 0;
      }
      else {
        for ( jc=0; jc<n; jc++ )
          for ( ic=0; ic<m; ic++ )
            Crow(ic,jc) = 0;
      }
    else
      if ( orderC=='C' ) {
        for ( jc=0; jc<n; jc++ )
          for ( ic=0; ic<m; ic++ )
            Ccol(ic,jc) = beta*Ccol(ic,jc);
      }
      else {
        for ( jc=0; jc<n; jc++ )
          for ( ic=0; ic<m; ic++ )
            Crow(ic,jc) = beta*Crow(ic,jc);
      }
    return;
  }

  if ( (transA=='N')&&(transB=='N') ) {
    for ( j=0; j<n; j++ ) 
      for ( i=0; i<m; i++ ) {
        tmp = 0; 
        if ( (orderA=='C')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(i,p) * Bcol(p,j);
        }
        else if ( (orderA=='C')&&(orderB=='R') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(i,p) * Brow(p,j);
        }
        else if ( (orderA=='R')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Arow(i,p) * Bcol(p,j);
        }
        else {
          for ( p=0; p<k; p++ )
            tmp += Arow(i,p) * Brow(p,j);
        }

	if ( beta==zero ) {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp;
          else
            Crow(i,j) = alpha*tmp;
        }
	else {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
          else
            Crow(i,j) = alpha*tmp + beta*Crow(i,j);
        }
      }
  }
  else if ( (transA=='N')&&(transB=='T') ) {
    for ( j=0; j<n; j++ ) 
      for ( i=0; i<m; i++ ) {
        tmp = 0; 
        if ( (orderA=='C')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(i,p) * Bcol(j,p);
        }
        else if ( (orderA=='C')&&(orderB=='R') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(i,p) * Brow(j,p);
        }
        else if ( (orderA=='R')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Arow(i,p) * Bcol(j,p);
        }
        else {
          for ( p=0; p<k; p++ )
            tmp += Arow(i,p) * Brow(j,p);
        }

	if ( beta==zero ) {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp;
          else
            Crow(i,j) = alpha*tmp;
        }
	else {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
          else
            Crow(i,j) = alpha*tmp + beta*Crow(i,j);
        }
      }
  }
  else if ( (transA=='T')&&(transB=='N') ) {
    for ( j=0; j<n; j++ ) 
      for ( i=0; i<m; i++ ) {
        tmp = 0; 
        if ( (orderA=='C')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(p,i) * Bcol(p,j);
        }
        else if ( (orderA=='C')&&(orderB=='R') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(p,i) * Brow(p,j);
        }
        else if ( (orderA=='R')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Arow(p,i) * Bcol(p,j);
        }
        else {
          for ( p=0; p<k; p++ )
            tmp += Arow(p,i) * Brow(p,j);
        }

	if ( beta==zero ) {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp;
          else
            Crow(i,j) = alpha*tmp;
        }
	else {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
          else
            Crow(i,j) = alpha*tmp + beta*Crow(i,j);
        }
      }
  }
  else if ( (transA=='T')&&(transB=='T') ) {
    for ( j=0; j<n; j++ ) 
      for ( i=0; i<m; i++ ) {
        tmp = 0; 
        if ( (orderA=='C')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(p,i) * Bcol(j,p);
        }
        else if ( (orderA=='C')&&(orderB=='R') ) {
          for ( p=0; p<k; p++ )
            tmp += Acol(p,i) * Brow(j,p);
        }
        else if ( (orderA=='R')&&(orderB=='C') ) {
          for ( p=0; p<k; p++ )
            tmp += Arow(p,i) * Bcol(j,p);
        }
        else {
          for ( p=0; p<k; p++ )
            tmp += Arow(p,i) * Brow(j,p);
        }

	if ( beta==zero ) {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp;
          else
            Crow(i,j) = alpha*tmp;
        }
	else {
          if ( orderC=='C' )
            Ccol(i,j) = alpha*tmp + beta*Ccol(i,j);
          else
            Crow(i,j) = alpha*tmp + beta*Crow(i,j);
        }
      }
  }
}

// ----------------------------------------------------------------------------
// BENCHMARK MAIN
// ----------------------------------------------------------------------------
#ifndef CONFIG_ITERATIONS
static const int N_REPS = 5;
#else
static const int N_REPS = CONFIG_ITERATIONS;
#endif

// What this run will do, printed once at the top: every setting in effect, including the
// ones nobody typed. `source` says where each came from -- "default", the flag or
// environment variable that set it, or config.json -- so a default is never invisible.
struct RunSettings {
    bool ref_protocol;
    string protocol_source;
    double tmin;
    string tmin_source;
    bool scalar_pack;
    string scalar_pack_source;
    bool zero_pack_lines;
    int pad_ldb;
    string pad_ldb_source;
    bool no_nc_overrides;
    string dataset;
    string raw_log, summary_log, perf_log;
    bool save_reps;
    string reps_log;
    vector<RawEvent> raw_events;
    bool raw_open;
};

static void overview_line(const char* key, const string& value, const string& source = "") {
    cout << "  " << left << setw(14) << key << (*key ? " : " : "   ") << value;
    if (!source.empty()) cout << "   [" << source << "]";
    cout << "\n";
}

static void print_run_overview(const RunSettings& r) {
    cout << "\n============================================================\n"
         << " Run overview   ([default] = not set by you)\n"
         << "============================================================\n";
#if RVV_INSTRUMENTED
    overview_line("Build", "INSTRUMENTED (RVV_INSTRUMENTED=1) -- DEBUG build",
                  "fixed at build time: make INSTRUMENTED=");
    overview_line("", "per-tile timers cost 11-17% of measured time: do NOT quote against the reference");
    const string protocol_note = r.protocol_source == "default"
        ? "default for this build: the reference protocol needs the uninstrumented build"
        : r.protocol_source;
#else
    overview_line("Build", "UNINSTRUMENTED (RVV_INSTRUMENTED=0) -- comparable to the reference design",
                  "fixed at build time: make INSTRUMENTED=");
    overview_line("", "per-tile timers and perf counters compiled out; pack/compute breakdown = nan");
    const string protocol_note = r.protocol_source;
#endif
    if (r.ref_protocol) {
        overview_line("Protocol", "REFERENCE (matches test_gemm.c:339-361)", protocol_note);
        overview_line("Warm-up", "none: the cold first rep is inside the mean");
        ostringstream reps;
        reps << "until tmin = " << r.tmin << " s has passed, one clock read per rep";
        overview_line("Reps", reps.str(), r.tmin_source);
    } else {
        overview_line("Protocol", "WARM-UP (stable numbers for short shapes; not the reference's protocol)",
                      protocol_note);
        overview_line("Warm-up", "5 kernel calls, then a scalar-pack pass of " + to_string(N_REPS) + " reps");
        overview_line("Reps", to_string(N_REPS) + " fixed reps", "iterations in config.json");
    }
#ifdef SKIP_REF_RUN
    overview_line("Correctness", "SKIPPED -- no scalar reference GEMM is run", "skip_ref_run in config.json");
#else
    overview_line("Correctness",
                  string("checked against the scalar reference GEMM, ") +
                      (r.ref_protocol ? "AFTER timing" : "BEFORE timing"),
                  "skip_ref_run is false in config.json");
#endif
    overview_line("Packing", r.scalar_pack ? "scalar packing forced" : "RVV vector packing",
                  r.scalar_pack ? r.scalar_pack_source : "default");
    // Issue #43: whether the pack-buffer lines are cleared (cbo.zero) before each pack.
    overview_line("Pack lines", r.zero_pack_lines
                      ? "cleared with cbo.zero before each pack (inside the pack timer)"
                      : "not cleared: each line the packer writes is read from the L2 first",
                  r.zero_pack_lines ? "--zero-pack-lines" : "default");
    // Issue #49: pads B's leading dimension (ldb = k + pad) to move where the B read
    // streams land in memory, without changing k, the data or the correctness check.
    overview_line("Pad ldb", r.pad_ldb > 0
                      ? ("ldb = k + " + to_string(r.pad_ldb) + " elements")
                      : "ldb = k (no padding)",
                  r.pad_ldb_source);
    overview_line("Kernel family", string(CONFIG_DATATYPE) + " " + CONFIG_METHOD + " | VLEN " +
                                       to_string(CONFIG_VLEN) + " | arch " + CONFIG_ARCH,
                  "config.json");
    string variants;
    for (int v = 0; v < num_variants; ++v) variants += (v ? ", " : "") + string(generated_variants[v].name);
    overview_line("Variants", variants, "config.json");
    overview_line("Cache blocking", "analytical model (get_optim_mc_nc_kc)", "default");
    string overrides;
    for (size_t i = 0; i < CONFIG_NC_OVERRIDES_COUNT; ++i)
        overrides += (i ? ", " : "") + string(CONFIG_NC_OVERRIDES[i].label) + " -> nc=" + to_string(CONFIG_NC_OVERRIDES[i].nc);
    if (overrides.empty()) overrides = "none";
    // Issue #16: the reference design has no per-shape nc override, so a head-to-head run
    // switches them off; the line still names what was switched off.
    if (r.no_nc_overrides)
        overview_line("", "nc overrides: OFF (every shape uses the heuristic's nc; config.json has: " + overrides + ")",
                      "--no-nc-overrides");
    else
        overview_line("", "nc overrides: " + overrides + " (applied to the row with exactly that label)", "config.json");
    const char* forced_nc = std::getenv("RVV_FORCE_NC");
    overview_line("", forced_nc ? string("RVV_FORCE_NC=") + forced_nc + " overrides nc for every shape"
                                : string("RVV_FORCE_NC: not set"),
                  forced_nc ? "environment" : "default");
    overview_line("Dataset", r.dataset);
    overview_line("Logs", r.raw_log);
    overview_line("", r.summary_log);
    overview_line("", r.perf_log);
    // Issues #35/#36: whether slow reps come in runs or at random needs each rep's own time.
    if (r.save_reps)
        overview_line("", r.reps_log + " (every timed rep, in the order it ran)", "--save-reps");
    else
        overview_line("", "per-rep times: not saved", "default");
    // Issue #42: the core's own event codes, counted per rep, in the per-rep log only.
    if (r.raw_events.empty()) {
        overview_line("", "raw events: none", "default");
    } else {
        string list;
        for (const RawEvent& e : r.raw_events) list += (list.empty() ? "" : ", ") + e.name + "=" + hex_code(e.code);
        overview_line("", "raw events: " + list + (r.raw_open
                              ? " -- counting; a Pack_ and a Compute_ column each in the per-rep log"
                              : " -- could not open: their per-rep columns are nan"),
                      "--raw-events");
    }
}

int main(int argc, char** argv) {
    bool force_scalar_pack = (getenv("USE_SCALAR_PACK") != NULL);
    string scalar_pack_source = force_scalar_pack ? "USE_SCALAR_PACK" : "default";
    bool no_nc_overrides = false;
    bool zero_pack_lines = false;  // issue #43
    int pad_ldb = 0;  // issue #49: elements added to B's leading dimension beyond k
    string pad_ldb_source = "default";
    // Issue #27: the reference design's run protocol (test_gemm.c:339-361) is the default
    // wherever it can run. The instrumented build cannot (its timers make the numbers
    // incomparable, issue #14), so its default is the older warm-up protocol.
#if RVV_INSTRUMENTED
    bool ref_protocol = false;
#else
    bool ref_protocol = true;
#endif
    string protocol_source = "default";
    double tmin = 3.0;  // seconds; matches run_ref_baseline.sh
    string tmin_source = "default";
    string dataset_file = "";
    bool save_reps = false;
    bool raw_events_given = false;
    string raw_events_arg;

    for (int i = 1; i < argc; ++i) {
        if (string(argv[i]) == "--scalar-pack") {
            force_scalar_pack = true;
            scalar_pack_source = "--scalar-pack";
        } else if (string(argv[i]) == "--zero-pack-lines") {
            zero_pack_lines = true;
        } else if (string(argv[i]) == "--pad-ldb") {
            char* end;
            pad_ldb = (i + 1 < argc) ? (int) std::strtol(argv[i + 1], &end, 10) : -1;
            if (i + 1 >= argc || end == argv[i + 1] || *end != '\0' || pad_ldb < 0) {
                cerr << "Error: --pad-ldb needs a non-negative integer number of elements." << endl;
                return 1;
            }
            pad_ldb_source = "--pad-ldb";
            ++i;
        } else if (string(argv[i]) == "--no-nc-overrides") {
            no_nc_overrides = true;
        } else if (string(argv[i]) == "--save-reps") {
            save_reps = true;
        } else if (string(argv[i]) == "--raw-events") {
            raw_events_given = true;
            if (i + 1 < argc) raw_events_arg = argv[++i];
        } else if (string(argv[i]) == "--ref-protocol") {
            ref_protocol = true;
            protocol_source = "--ref-protocol";
        } else if (string(argv[i]) == "--no-ref-protocol") {
            ref_protocol = false;
            protocol_source = "--no-ref-protocol";
        } else if (string(argv[i]) == "--tmin") {
            char* end = nullptr;
            tmin = (i + 1 < argc) ? std::strtod(argv[i + 1], &end) : 0.0;
            if (i + 1 >= argc || end == argv[i + 1] || *end != '\0' || !(tmin > 0.0)) {
                cerr << "Error: --tmin needs a positive number of seconds." << endl;
                return 1;
            }
            tmin_source = "--tmin";
            ++i;
        } else {
            dataset_file = argv[i];
        }
    }

    if (dataset_file == "") {
        cerr << "Usage: " << argv[0] << " <dataset_file.dat> [--scalar-pack] [--zero-pack-lines] [--pad-ldb <elements>] [--no-ref-protocol | --ref-protocol] [--tmin <seconds>] [--no-nc-overrides] [--save-reps] [--raw-events <name=code,...|x60-l2>]" << endl;
        return 1;
    }
    // Issue #43: the DOT dispatch clears no lines (its K pad is zeroed on purpose instead), so a
    // DOT run with the switch would carry a name for something it did not do.
    if (zero_pack_lines && string(CONFIG_METHOD).find("DOT") != string::npos) {
        cerr << "Error: --zero-pack-lines is for the AXPY kernels only; this is a DOT kernel ("
             << CONFIG_METHOD << "), whose dispatch clears no lines." << endl;
        return 1;
    }
    vector<RawEvent> raw_events;
    if (raw_events_given) {
#if !RVV_INSTRUMENTED
        cerr << "Error: --raw-events needs the instrumented build (make INSTRUMENTED=1): this build compiles\n"
                "       the counters out, so that its times carry no counter reads (issue #14)." << endl;
        return 1;
#endif
        string why = parse_raw_events(raw_events_arg, raw_events);
        if (!why.empty()) {
            cerr << "Error: " << why << endl;
            return 1;
        }
        if (!save_reps) {
            cerr << "Error: --raw-events needs --save-reps: its counts are written only to the per-rep log." << endl;
            return 1;
        }
    }
    if (tmin_source != "default" && !ref_protocol) {
        cerr << "Error: --tmin only applies to the reference protocol (this run uses the warm-up protocol:\n"
                "       " << (protocol_source == "default" ? "the instrumented build's default" : "--no-ref-protocol") << ")." << endl;
        return 1;
    }
#if RVV_INSTRUMENTED
    if (ref_protocol) {
        cerr << "Error: --ref-protocol needs the uninstrumented build (make INSTRUMENTED=0).\n"
                "       The instrumented build's timers cost 11-17% of measured time, so its\n"
                "       numbers are not comparable to the reference design under any protocol (issue #14)." << endl;
        return 1;
    }
#endif
    // Logs of the two protocols never share a file, so one can never be merged into the other.
    // Likewise a run without the config.json nc overrides never shares a file with one that
    // has them: they are different configurations (issue #16).
    // A scalar-pack-live run is the issue #26 experiment, so it never shares a log with the
    // vector-pack one either, nor a run that clears the pack lines (issue #43) one that does not.
    const string log_suffix = string(RVV_LOG_SUFFIX) + (ref_protocol ? "_refproto" : "") +
                              (force_scalar_pack ? "_scalarpack" : "") +
                              (zero_pack_lines ? "_zeroline" : "") +
                              (pad_ldb > 0 ? ("_padldb" + to_string(pad_ldb)) : "") +
                              (no_nc_overrides ? "_no_nc_override" : "");
    const string log_stem = string("_") + CONFIG_DATATYPE + "_" + CONFIG_METHOD + log_suffix + ".txt";
    // Ensure logs directory exists (handled by Makefile)
    string raw_log_name = "logs/raw" + log_stem;
    string summary_log_name = "logs/summary" + log_stem;
    string perf_log_name = "logs/perf" + log_stem;
    string reps_log_name = "logs/reps" + log_stem;
    // Issue #8: hardware counters supplement the wall-clock columns; they go to a separate
    // log so the raw log's schema is unchanged. Opened before the overview so that it can say
    // whether the raw events (issue #42) opened.
    perf_gemm_counters perf;
    perf_gemm_open(&perf);
    bool raw_open = false;
    if (!raw_events.empty()) {
        vector<uint64_t> codes;
        for (const RawEvent& e : raw_events) codes.push_back(e.code);
        raw_open = perf_gemm_open_raw(&perf, codes.data(), (int) codes.size()) == 0;
    }
    print_run_overview({ref_protocol, protocol_source, tmin, tmin_source, force_scalar_pack,
                        scalar_pack_source, zero_pack_lines, pad_ldb, pad_ldb_source, no_nc_overrides, dataset_file, raw_log_name, summary_log_name, perf_log_name,
                        save_reps, reps_log_name, raw_events, raw_open});
    bool raw_exists = false;
    {
        ifstream check(raw_log_name);
        if (check.good() && check.peek() != std::ifstream::traits_type::eof()) {
            raw_exists = true;
        }
    }
    ofstream raw_log(raw_log_name, ios_base::app);
    if (!raw_log.is_open()) {
        cerr << "Warning: Could not open " << raw_log_name << " for writing." << endl;
    } else if (!raw_exists) {
        raw_log << "Layer,M,N,K,Variant,Status,KerGOPs_Harmonic,KerGOPs_Min,KerGOPs_P5,KerGOPs_P95,KerGOPs_Max,KerGOPs_IQR,KerGOPs_Mean,KerGOPs_Std,KerGOPs_Median,RVV_Pack_GBs,Scl_Pack_GBs,RVV_Time_ms,Ref_Time_ms,Vector_Pack_Time_ms,Scalar_Pack_Time_ms,Compute_Time_ms,N_Reps\n";
    }

    ofstream summary_log(summary_log_name, ios_base::app);
    if (!summary_log.is_open()) {
        cerr << "Warning: Could not open " << summary_log_name << " for writing." << endl;
    }

#if RVV_INSTRUMENTED
    if (perf.fd < 0) {
        cout << "[INFO] perf_event_open() counter group unavailable; perf_ log columns will be nan." << endl;
    }
#endif
    bool perf_exists = false;
    {
        ifstream check(perf_log_name);
        perf_exists = check.good() && check.peek() != std::ifstream::traits_type::eof();
    }
    ofstream perf_log(perf_log_name, ios_base::app);
    if (!perf_log.is_open()) {
        cerr << "Warning: Could not open " << perf_log_name << " for writing." << endl;
    } else if (!perf_exists) {
        perf_log << "Layer,M,N,K,Variant,Status,Counters_Valid";
        for (const char* region : PERF_REGIONS) {
            for (const char* metric : {"Cycles", "Instructions", "IPC", "L1D_Read_Miss_Pct", "L1D_Write_Miss_Pct", "dTLB_Misses",
                                       "Stall_BE_Pct", "Stall_FE_Pct", "Branch_Misses"}) {
                perf_log << "," << region << "_" << metric;
            }
        }
        perf_log << ",Vector_Pack_Cycles_CV_Pct,Compute_Cycles_CV_Pct,Compute_FLOPs_per_Cycle,Arith_Intensity_FLOPs_per_Byte\n";
    }

    // One row per timed rep. Written after each shape's timed loop, from the times the
    // loop already keeps, so saving them adds nothing inside the timed interval.
    ofstream reps_log;
    if (save_reps) {
        bool reps_exists = false;
        {
            ifstream check(reps_log_name);
            reps_exists = check.good() && check.peek() != std::ifstream::traits_type::eof();
        }
        reps_log.open(reps_log_name, ios_base::app);
        if (!reps_log.is_open()) {
            cerr << "Warning: Could not open " << reps_log_name << " for writing." << endl;
        } else if (!reps_exists) {
            reps_log << "Layer,M,N,K,Variant,MC,NC,KC,Rep,Seconds,Pack_Seconds,Compute_Seconds,Pack_Cycles,Compute_Cycles";
            for (const RawEvent& e : raw_events) reps_log << ",Pack_" << e.name << ",Compute_" << e.name;
            reps_log << "\n";
        }
    }

    ifstream dataset_stream(dataset_file);
    if (!dataset_stream.is_open()) {
        cerr << "Error: Could not open dataset file " << dataset_file << endl;
        return 1;
    }
    cout << "\n============================================================" << endl;
    cout << " Benchmark Config: Arch=" << CONFIG_ARCH 
         << " | Datatype=" << CONFIG_DATATYPE 
         << " | Method=" << CONFIG_METHOD 
         << " | VLEN=" << CONFIG_VLEN << endl;
    cout << " Dataset: " << dataset_file << endl;
    cout << "============================================================\n" << endl;

    cout << " How to read this output\n";
    cout << "   GOPs   [billions of operations per second = 2*M*N*K / time; a multiply-add counts as 2; higher is better]\n";
    cout << "   Timed  [" << TIMED_VALUE << " -- " << TIMED_NOTE << "]\n";
    cout << "   Spread [each rep's own GOPs: the slowest, 5th percentile, 95th percentile and fastest rep, the width of the middle 50%,\n";
    cout << "           the standard deviation, and the fastest-to-median ratio; a wide spread means unstable timing]\n";
    cout << "   Full definitions of every log column: docs/metrics.md\n";
    cout << "\n" << endl;

    string line;
    while (getline(dataset_stream, line)) {
        if (line.empty() || line[0] == '#') continue;

        istringstream iss(line);
        string label;
        int m, n, k;
        if (!(iss >> label >> m >> n >> k)) {
            continue; 
        }

        // Issue #49: ldb (B's leading dimension, column-major) is k plus an optional pad in
        // elements (--pad-ldb), to move where the NR read streams land in memory without
        // changing k, the data or the correctness check below.
        int lda = m, ldb = k + pad_ldb, ldc = m;

        // Over-allocate buffers by 32 in each dimension to prevent segfaults/heap corruption.
        // The simple pure_xdsl macrokernel does not have edge-case handling for un-padded fringes,
        // so it will unconditionally write out of bounds up to the next MR/NR multiple.
        size_t pad_m = m + 32;
        size_t pad_n = n + 32;
        size_t pad_k = k + 32;
        size_t pad_ldb_rows = (size_t) ldb + 32;  // B's row count follows ldb, not k, once padded
        A_TYPE* A = alloc_aligned<A_TYPE>(pad_m * pad_k);
        B_TYPE* B = alloc_aligned<B_TYPE>(pad_ldb_rows * pad_n);
        C_TYPE* C = alloc_aligned<C_TYPE>(pad_m * pad_n);
        C_TYPE* Cg = alloc_aligned<C_TYPE>(pad_m * pad_n);
        
        fill_random(A, lda * k);
        fill_random(B, ldb * n);
        
        float alpha = 1.0f;
        float beta = 0.0f;

        string ref_time_str = "N/A";
        double ref_time_val_global = 0.0;
        string dim_str = to_string(m) + "x" + to_string(n) + "x" + to_string(k);
        // Bounded reference check (skip if too massive)
#ifdef SKIP_REF_RUN
        bool do_ref = false;
#else
        bool do_ref = true;
#endif
        if (do_ref) {
            cout << "\r" << "  [" << label << " (" << dim_str << ")] Running scalar reference ... " << flush;
            for (size_t i = 0; i < (size_t)(ldc * n); ++i) Cg[i] = 0;
            auto ref_start = chrono::high_resolution_clock::now();
            gemm_ref('C', 'C', 'C', 'N', 'N', m, n, k, alpha, A, lda, B, ldb, beta, Cg, ldc);
            auto ref_end = chrono::high_resolution_clock::now();
            chrono::duration<double> ref_diff = ref_end - ref_start;
            ref_time_val_global = ref_diff.count() * 1000.0;
            
            stringstream ss;
            ss << fixed << setprecision(4) << ref_time_val_global;
            ref_time_str = ss.str();
        }

        struct VariantResult {
            string name;
            string status;
            double ker_gflops;
            double ker_gflops_min;
            double ker_gflops_p5;
            double ker_gflops_p95;
            double ker_gflops_max;
            double ker_gflops_iqr;
            double ker_gflops_mean;
            double ker_gflops_std;
            double ker_gflops_median;
            double rvv_bw;
            double scl_bw;
            double rvv_time_ms;
            double ref_time_ms;
            double vector_pack_time_ms;
            double scalar_pack_time_ms;
            double compute_time_ms;
            double sample_rvv;
            double sample_ref;
            int sample_i;
            int sample_j;
            int mc;
            int nc;
            int kc;
            double compute_ipc;
            double e2e_gops;  // 2*M*N*K / mean time per rep: the reference design's own formula
            int n_reps;
            int mr;
            int nr;
        };
        vector<VariantResult> variant_results;

        for (int v = 0; v < num_variants; ++v) {
            const auto& variant = generated_variants[v];

            int mc, nc, kc;
            get_optim_mc_nc_kc(sizeof(float), m, n, k, variant.mr, variant.nr, &mc, &nc, &kc);

            // Debug/mitigation override for issue #1 (B3_M7 packing instability): apply a
            // per-shape nc override from config.json to the row with exactly that label,
            // without changing get_optim_mc_nc_kc's general heuristic for any other shape.
            // The reference design has no such override, so --no-nc-overrides skips it (issue #16).
            if (!no_nc_overrides) {
                for (size_t oi = 0; oi < CONFIG_NC_OVERRIDES_COUNT; ++oi) {
                    const auto& ov = CONFIG_NC_OVERRIDES[oi];
                    if (label == ov.label) {
                        nc = (n < ov.nc) ? n : ov.nc;
                        break;
                    }
                }
            }

            // Ad hoc override for quick interactive experiments; takes priority over config.json.
            if (const char* forced_nc = std::getenv("RVV_FORCE_NC")) {
                int nc_req = std::atoi(forced_nc);
                nc = (n < nc_req) ? n : nc_req;
            }

            int kc_pad = kc;
            if (string(CONFIG_METHOD).find("DOT") != string::npos) {
                kc_pad = ((kc + 3) / 4) * 4;
            }

            size_t padded_mc = ((mc + variant.mr - 1) / variant.mr) * variant.mr;
            size_t padded_nc = ((nc + variant.nr - 1) / variant.nr) * variant.nr;
            // The reference's sizes (test_gemm.c:271-272, built with -DKR=1): Ac holds
            // (MR+mc)*(1+kc) floats and Bc (1+kc)*(NR+nc), not (mr+kc) / (nr+kc) (issue #19).
            // The panels the packers fill are never larger, so the floor below only matters for
            // DOT, whose K padding (kc_pad) can exceed the reference's one spare row.
            size_t ac_alloc_size = std::max<size_t>((size_t)(variant.mr + mc) * (1 + kc), padded_mc * kc_pad);
            size_t bc_alloc_size = std::max<size_t>((size_t)(1 + kc) * (variant.nr + nc), padded_nc * kc_pad);
            A_TYPE* Ac = alloc_aligned_zeroed<A_TYPE>(ac_alloc_size);
            B_TYPE* Bc = alloc_aligned_zeroed<B_TYPE>(bc_alloc_size);
     // Re-initialize C for this variant
            for (size_t i = 0; i < (size_t)(ldc * n); ++i) C[i] = 0;

            // Warmup: run multiple times to bring A, B, Ac, Bc hot into cache
            // A single warmup run is insufficient for large working sets.
            // Also what fills C for the correctness check below. Under --ref-protocol
            // (issue #27) there is no warm-up at all, so the check moves after the timed
            // loop, as in test_gemm.c, and reads the C that loop left behind.
            if (!ref_protocol) {
                for (int w = 0; w < 5; ++w) {
                    double dump_pt=0, dump_kt=0; uint64_t dump_pb=0;
                    variant.func(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, Ac, Bc, mc, nc, kc, variant.mr, variant.nr, &dump_pt, &dump_kt, &dump_pb, force_scalar_pack, zero_pack_lines, nullptr);
                }
            }

            string status = "SKIP";
            double sample_rvv = 0.0;
            double sample_ref = 0.0;
            int sample_i = rand() % m;
            int sample_j = rand() % n;

            auto check_against_reference = [&]() {
                sample_rvv = C[sample_j * ldc + sample_i];
                sample_ref = Cg[sample_j * ldc + sample_i];

                double error = 0.0;
                double nrm = 0.0;
                double tmp = 0.0;
                int mismatched = 0;
                // Legacy error computation
                for (size_t j = 0; j < (size_t)n; ++j) {
                    for (size_t i = 0; i < (size_t)m; ++i) {
                        double ref_val = (double) Cg[j * ldc + i];
                        nrm += ref_val * ref_val;
                        tmp = (double) std::abs(ref_val - C[j * ldc + i]);
                        error += tmp * tmp;
                        
                        // Track first few mismatches for debugging
                        if (tmp > 1e-4 && mismatched < 5) {
                            if (mismatched == 0) cout << endl << "  [MISMATCH DETECTED] " << variant.name << endl;
                            cout << "    C[" << i << "," << j << "] -> Ref: " << Cg[j * ldc + i] << " | RVV: " << C[j * ldc + i] << " | Diff: " << tmp << endl;
                            mismatched++;
                        }
                    }
                }
                if (nrm != 0.0) {
                    error = sqrt(error) / sqrt(nrm);
                } else {
                    error = sqrt(error);
                }
                
                double tol = GET_ERR_TOL(k);
                if (error > tol) status = "FAIL";
                else status = "PASS";
            };
            if (do_ref && !ref_protocol) check_against_reference();

            string current_ref_time = ref_time_str;
            string label_str = label;
            string var_str = string("`") + variant.name + "`";

            cout << "\r" << "  [" << label << " (" << dim_str << ")] Testing variant " << (v+1) << "/" << num_variants << " : " << left << setw(15) << variant.name << flush;
            
            double ker_gflops_val = 0.0;
            double ker_gflops_min = 0.0;
            double ker_gflops_p5 = 0.0;
            double ker_gflops_p95 = 0.0;
            double ker_gflops_max = 0.0;
            double ker_gflops_iqr = 0.0;
            double ker_gflops_mean = 0.0;
            double ker_gflops_std = 0.0;
            double ker_gflops_median = 0.0;
            double rvv_bw_val = 0.0;
            double scl_bw_val = 0.0;
            double rvv_time_val = 0.0;
            double ref_time_val = ref_time_val_global;
            double vector_pack_time_val = 0.0;
            double scalar_pack_time_val = 0.0;
            double compute_time_val = 0.0;
            bool scalar_perf_valid = false, rvv_perf_valid = false;
            perf_counts scalar_pack_counts = {}, vector_pack_counts = {}, compute_counts = {};
            vector<double> rep_pack_cycles, rep_compute_cycles;
            // Issue #42: each rep's raw counts, n_raw per rep; nan for a rep any of whose
            // windows the raw group was shared out for, or once a read of it failed.
            const size_t n_raw = raw_events.size();
            vector<double> rep_raw_pack, rep_raw_compute;
            double num_ops = 2.0 * m * n * k;
            double pack_bytes_per_rep = 0.0;
            // Reps the RVV loop actually ran; 0 if it did not run (default protocol, FAIL).
            int n_reps = 0;

            if (ref_protocol || status != "FAIL") {
                uint64_t total_pack_bytes = 0;
                // Benchmark Scalar Packing
                // Reset C to zero so each benchmark starts from a consistent state
                for (size_t i = 0; i < (size_t)(ldc * n); ++i) C[i] = 0;
                double total_scalar_pack_time = 0.0;
                perf_gemm_reset(&perf);
                // This pass runs in BOTH builds even though an uninstrumented one can
                // measure nothing from it (issue #14). It is the RVV loop's de facto
                // warm-up -- it first-touches C, warms Ac/Bc (already page-touched: they are
                // allocated zeroed, issue #18) and trains the predictors -- and
                // dropping it made short shapes (B3) come out ~14% SLOWER uninstrumented,
                // because one cold rep then dominated the mean. The two builds have to
                // differ in the timers and nothing else, or the A/B means nothing.
                // Not under --ref-protocol (issue #27): the reference has no such pass.
                for (int r = 0; !ref_protocol && r < N_REPS; ++r) {
                    double pt = 0.0, kt = 0.0; uint64_t pb = 0;
                    variant.func(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, Ac, Bc, mc, nc, kc, variant.mr, variant.nr, &pt, &kt, &pb, true, zero_pack_lines, &perf);
                    total_scalar_pack_time += pt;
                    total_pack_bytes += pb;
                }
                if (!ref_protocol) {
                    scalar_pack_counts = perf.pack;
                    scalar_perf_valid = perf.valid;
                }
#if RVV_INSTRUMENTED
                scl_bw_val = (double)total_pack_bytes / (total_scalar_pack_time * 1e9);
                // Issue #7: this pass already runs to produce Scl_Pack_GBs above; expose
                // its per-rep-averaged ms directly instead of adding a second timing pass.
                scalar_pack_time_val = (total_scalar_pack_time / N_REPS) * 1000.0;
#else
                scl_bw_val = std::nan("");
                scalar_pack_time_val = std::nan("");
#endif

                // Benchmark RVV
                // Reset C again before RVV loop so it starts from the same clean state
                for (size_t i = 0; i < (size_t)(ldc * n); ++i) C[i] = 0;
                double total_rvv_pack_time = 0.0;
                double total_kernel_time = 0.0;
                total_pack_bytes = 0;
                // Seconds charged to each rep, and the denominator of every KerGOPs
                // figure below. Instrumented: the inner pack+compute timers, as issue #13
                // settled. Uninstrumented: this rep's own wall clock (issue #14).
                // Reserved up front so no push_back reallocates inside the timed loop:
                // a reallocation would land in one rep's interval, and only under
                // --ref-protocol, whose rep count is not known in advance.
                std::vector<double> rep_total;
                rep_total.reserve(ref_protocol ? (size_t)1 << 20 : (size_t)N_REPS);
#if RVV_INSTRUMENTED
                // The two halves of rep_total, kept apart for --save-reps.
                std::vector<double> rep_pack_s, rep_compute_s;
                rep_pack_s.reserve(N_REPS);
                rep_compute_s.reserve(N_REPS);
#endif
                perf_gemm_reset(&perf);

                auto start = chrono::high_resolution_clock::now();
#if !RVV_INSTRUMENTED
                // The reference reads its clock ONCE per rep and differences against a
                // single t1 taken before the loop (test_gemm.c:339-361):
                //     t1 = dclock(); while (time <= tmin) { gemm(...); t2 = dclock(); ... }
                // `start` is our t1. Differencing consecutive marks gives per-rep times
                // at the same one-read-per-rep cost, rather than a pair per rep.
                auto prev_mark = start;
#endif
                // Default: N_REPS reps. --ref-protocol (issue #27): the reference's
                // `while (time <= tmin)` -- stop at the first rep whose end mark is past
                // tmin. Both run the same body; only the stop test differs.
                for (;;) {
                    double pt = 0.0, kt = 0.0; uint64_t pb = 0;
#if RVV_INSTRUMENTED
                    uint64_t pack_cycles_before = perf.pack.v[PERF_EV_CYCLES];
                    uint64_t compute_cycles_before = perf.compute.v[PERF_EV_CYCLES];
                    const perf_counts raw_pack_before = perf.pack, raw_compute_before = perf.compute;
                    const uint64_t raw_rejected_before = perf.raw_rejected;
#endif
                    variant.func(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, Ac, Bc, mc, nc, kc, variant.mr, variant.nr, &pt, &kt, &pb, force_scalar_pack, zero_pack_lines, &perf);
#if RVV_INSTRUMENTED
                    rep_total.push_back(kt + pt);
                    rep_pack_s.push_back(pt);
                    rep_compute_s.push_back(kt);
                    rep_pack_cycles.push_back((double) (perf.pack.v[PERF_EV_CYCLES] - pack_cycles_before));
                    rep_compute_cycles.push_back((double) (perf.compute.v[PERF_EV_CYCLES] - compute_cycles_before));
                    const bool raw_ok = perf.raw_valid && perf.raw_rejected == raw_rejected_before;
                    for (size_t e = 0; e < n_raw; ++e) {
                        rep_raw_pack.push_back(raw_ok ? (double) (perf.pack.raw[e] - raw_pack_before.raw[e]) : std::nan(""));
                        rep_raw_compute.push_back(raw_ok ? (double) (perf.compute.raw[e] - raw_compute_before.raw[e]) : std::nan(""));
                    }
#else
                    auto rep_mark = chrono::high_resolution_clock::now();
                    rep_total.push_back(chrono::duration<double>(rep_mark - prev_mark).count());
                    prev_mark = rep_mark;
#endif
                    total_rvv_pack_time += pt;
                    total_kernel_time += kt;
                    total_pack_bytes += pb;
                    ++n_reps;
#if !RVV_INSTRUMENTED
                    if (ref_protocol) {
                        if (chrono::duration<double>(rep_mark - start).count() > tmin) break;
                        continue;
                    }
#endif
                    if (n_reps >= N_REPS) break;
                }
                auto end = chrono::high_resolution_clock::now();
                vector_pack_counts = perf.pack;
                compute_counts = perf.compute;
                rvv_perf_valid = perf.valid;
                pack_bytes_per_rep = (double) total_pack_bytes / n_reps;

                chrono::duration<double> diff = end - start;
                double avg_time = diff.count() / n_reps;

                double sum_rep_total = 0.0;
                for (int r = 0; r < n_reps; ++r) sum_rep_total += rep_total[r];
                ker_gflops_val = (num_ops * n_reps) / (sum_rep_total * 1e9);

                // Compute per-rep stats
                std::vector<double> rep_gops(n_reps);
                double sum_gops = 0.0;
                for (int r = 0; r < n_reps; ++r) {
                    rep_gops[r] = num_ops / (rep_total[r] * 1e9);
                    sum_gops += rep_gops[r];
                }
                std::sort(rep_gops.begin(), rep_gops.end());
                ker_gflops_min = rep_gops.front();
                ker_gflops_p5 = rep_gops[(int)(0.05 * n_reps)];
                ker_gflops_median = rep_gops[n_reps / 2];
                ker_gflops_p95 = rep_gops[(int)(0.95 * n_reps)];
                ker_gflops_max = rep_gops.back();
                ker_gflops_iqr = rep_gops[(int)(0.75 * n_reps)] - rep_gops[(int)(0.25 * n_reps)];
                
                double mean_gops = sum_gops / n_reps;
                ker_gflops_mean = mean_gops;
                
                double sq_sum = 0.0;
                for (int r = 0; r < n_reps; ++r) {
                    sq_sum += (rep_gops[r] - mean_gops) * (rep_gops[r] - mean_gops);
                }
                ker_gflops_std = std::sqrt(sq_sum / n_reps);

                if (reps_log.is_open()) {
                    for (int r = 0; r < n_reps; ++r) {
                        reps_log << label_str << "," << m << "," << n << "," << k << "," << variant.name << ","
                                 << mc << "," << nc << "," << kc << "," << r << "," << rep_total[r] << ",";
#if RVV_INSTRUMENTED
                        reps_log << rep_pack_s[r] << "," << rep_compute_s[r] << ",";
                        if (perf.valid) reps_log << rep_pack_cycles[r] << "," << rep_compute_cycles[r];
                        else reps_log << "nan,nan";
                        for (size_t e = 0; e < n_raw; ++e)
                            reps_log << "," << rep_raw_pack[r * n_raw + e] << "," << rep_raw_compute[r * n_raw + e];
#else
                        reps_log << "nan,nan,nan,nan";
#endif
                        reps_log << "\n";
                    }
                    reps_log.flush();
                }

                rvv_time_val = avg_time * 1000.0;

#if RVV_INSTRUMENTED
                rvv_bw_val = (double)total_pack_bytes / (total_rvv_pack_time * 1e9);

                // Isolated pack-vs-compute breakdown (issue #2): per-rep-averaged
                // ms, separate from rvv_time_val (combined wall-clock incl. loop
                // overhead) and rvv_bw_val (pack-only throughput in GB/s, not time).
                vector_pack_time_val = (total_rvv_pack_time / n_reps) * 1000.0;
                compute_time_val = (total_kernel_time / n_reps) * 1000.0;
#else
                // Issue #14: no inner timers exist in this build, so there is no
                // breakdown to report. RVV_Time_ms above is the whole result.
                rvv_bw_val = std::nan("");
                vector_pack_time_val = std::nan("");
                compute_time_val = std::nan("");
#endif
            }

            if (ref_protocol) {
                // At tmin=3.0 the largest shapes finish in one or two reps, so the
                // reported number is a single cold sample. gen_ref_datasets.py warns
                // about the same thing on the reference side.
                if (n_reps < 3) {
                    cout << "\r                                                                               \r"
                         << "  [WARN] " << label << " " << variant.name << ": only " << n_reps
                         << " rep(s) fit in tmin=" << tmin << " s; the number is a near-single cold sample." << endl;
                }
                if (do_ref) check_against_reference();
            }

            // Per-rep denominators. A rep count of 0 (default protocol, FAILed before
            // timing) has no counts to divide, so fall back to the configured count.
            const int metric_reps = n_reps > 0 ? n_reps : N_REPS;
            perf_metrics vector_pack_metrics = perf_metrics_per_rep(&vector_pack_counts, metric_reps);
            perf_metrics scalar_pack_metrics = perf_metrics_per_rep(&scalar_pack_counts, N_REPS);
            perf_metrics compute_metrics = perf_metrics_per_rep(&compute_counts, metric_reps);
            double compute_ipc = rvv_perf_valid ? compute_metrics.ipc : std::nan("");

            variant_results.push_back({string(variant.name), status, ker_gflops_val, ker_gflops_min, ker_gflops_p5, ker_gflops_p95, ker_gflops_max, ker_gflops_iqr, ker_gflops_mean, ker_gflops_std, ker_gflops_median, rvv_bw_val, scl_bw_val, rvv_time_val, ref_time_val, vector_pack_time_val, scalar_pack_time_val, compute_time_val, sample_rvv, sample_ref, sample_i, sample_j, mc, nc, kc, compute_ipc,
                                     rvv_time_val > 0 ? num_ops / (rvv_time_val * 1e6) : std::nan(""), n_reps, variant.mr, variant.nr});

            if (perf_log.is_open()) {
                double nan_v = std::nan("");
                perf_log << label_str << "," << m << "," << n << "," << k << ","
                         << variant.name << "," << status << "," << (scalar_perf_valid && rvv_perf_valid ? 1 : 0);
                write_perf_metrics(perf_log, vector_pack_metrics, rvv_perf_valid);
                write_perf_metrics(perf_log, scalar_pack_metrics, scalar_perf_valid);
                write_perf_metrics(perf_log, compute_metrics, rvv_perf_valid);
                perf_log << "," << (rvv_perf_valid ? cv_pct_of(rep_pack_cycles) : nan_v)
                         << "," << (rvv_perf_valid ? cv_pct_of(rep_compute_cycles) : nan_v)
                         << "," << (rvv_perf_valid && compute_metrics.cycles > 0 ? num_ops / compute_metrics.cycles : nan_v)
                         << "," << (pack_bytes_per_rep > 0 ? num_ops / pack_bytes_per_rep : nan_v) << endl;
            }

            if (raw_log.is_open()) {
                raw_log << label_str << "," << m << "," << n << "," << k << ","
                        << variant.name << "," << status << ","
                        << ker_gflops_val << ","
                        << ker_gflops_min << "," << ker_gflops_p5 << ","
                        << ker_gflops_p95 << "," << ker_gflops_max << ","
                        << ker_gflops_iqr << ","
                        << ker_gflops_mean << "," << ker_gflops_std << ","
                        << ker_gflops_median << ","
                        << rvv_bw_val << "," << scl_bw_val << ","
                        << rvv_time_val << "," << ref_time_val << ","
                        << vector_pack_time_val << "," << scalar_pack_time_val << "," << compute_time_val << ","
                        << n_reps << endl;
            }

            // Live per-variant console block (issue #9), reworded so each figure says what it
            // measures. Only real numbers are printed: the uninstrumented build has no
            // pack/compute split and no counters, so it says so in one line instead of
            // printing rows of nan.
            if (status != "FAIL") {
                const double e2e_gops = rvv_time_val > 0 ? num_ops / (rvv_time_val * 1e6) : std::nan("");

                cout << "\r                                                                               \r";
                cout << "  +- " << label << " (" << dim_str << ") " << short_variant_name(variant.name)
                     << " [" << variant.name << ": " << variant.mr << " rows x " << variant.nr
                     << " columns of C per microkernel tile]\n";
                cout << "  |  Speed [GOPs = billions of operations per second = 2*M*N*K / time; higher is better]\n";
                report_row("by mean time", num(e2e_gops, 2) + " GOPs",
                           "mean of " + to_string(n_reps) + " reps = " + num(rvv_time_val, 3) +
                           " ms per GEMM call" + MEAN_TIME_EXTRA + "; the reference design's own formula");
                report_row("by median rep", num(ker_gflops_median, 2) + " GOPs",
                           "median of the per-rep GOPs; not thrown off by a few slow reps");
                report_row("what is timed", TIMED_VALUE, TIMED_NOTE);
                report_row("spread per rep",
                           "slowest " + num(ker_gflops_min, 2) + " | 5th percentile " + num(ker_gflops_p5, 2) +
                           " | 95th percentile " + num(ker_gflops_p95, 2) + " | fastest " + num(ker_gflops_max, 2),
                           "GOPs of single reps");
                const double cv_pct = ker_gflops_median > 0 ? 100.0 * ker_gflops_std / ker_gflops_median : 0.0;
                const double max_over_median = ker_gflops_median > 0 ? ker_gflops_max / ker_gflops_median : 0.0;
                report_row("",
                           "middle-50% width " + num(ker_gflops_iqr, 2) + " | std-dev " + num(ker_gflops_std, 2) +
                           " | std-dev / median " + num(cv_pct, 0) + "% [above 15% = unstable] | fastest / median " +
                           num(max_over_median, 2) + "x [above 1.25 = cache thrashing]");
                ostringstream reps_note;
                if (ref_protocol) reps_note << "until tmin=" << tmin << " s, reference protocol";
                else reps_note << "fixed, from config.json";
                report_row("reps", to_string(n_reps), reps_note.str());
                if (ref_time_val > 0) {
                    const double ref_gops = num_ops / (ref_time_val / 1000.0) / 1e9;
                    report_row("naive scalar GEMM", num(ref_gops, 2) + " GOPs",
                               "our own slow triple loop, run once to check the answer; NOT the reference design");
                    report_row("", "this run is " + num(e2e_gops / ref_gops, 1) + " times faster than it");
                } else {
                    report_row("naive scalar GEMM", "not run", "skip_ref_run in config.json");
                }
#if RVV_INSTRUMENTED
                const bool counters_valid = scalar_perf_valid && rvv_perf_valid;
                const double compute_flops_per_cycle = compute_metrics.cycles > 0 ? num_ops / compute_metrics.cycles : std::nan("");
                cout << "  |  Time split per rep [pack = reorder A and B into contiguous buffers; compute = microkernel loops]\n";
                report_row("vector pack", num(vector_pack_time_val, 2) + " ms (" +
                           pct_of_total(vector_pack_time_val, compute_time_val) + " of pack + compute)",
                           "the RVV vector-instruction packer");
                report_row("scalar pack", num(scalar_pack_time_val, 2) + " ms (" +
                           pct_of_total(scalar_pack_time_val, compute_time_val) + " of pack + compute)",
                           "the scalar packer, measured in its own pass");
                report_row("compute", num(compute_time_val, 2) + " ms", "microkernel loops");
                report_row("vector-pack gain", gain_pct(scalar_pack_time_val, vector_pack_time_val),
                           "how much faster the vector packer is than the scalar one");
                if (counters_valid) {
                    cout << "  |  Hardware counters [IPC = instructions per cycle; L1D miss = L1 data-cache miss rate on reads / writes;\n"
                            "  |                     back-end stall = share of cycles waiting on the pipeline back end; FLOPs/cycle = operations per cycle]\n";
                    auto counters_row = [&](const char* region, const perf_metrics& m, bool valid, const string& extra) {
                        report_row(region, "IPC " + perf_field(valid, m.ipc, 2, "") +
                                   " | L1D miss " + perf_field(valid, m.l1d_read_miss_pct, 1, "%") + " read / " +
                                   perf_field(valid, m.l1d_write_miss_pct, 1, "%") + " write | back-end stall " +
                                   perf_field(valid, m.stall_be_pct, 1, "%") + extra);
                    };
                    counters_row("vector pack", vector_pack_metrics, rvv_perf_valid, "");
                    counters_row("scalar pack", scalar_pack_metrics, scalar_perf_valid, "");
                    counters_row("compute", compute_metrics, rvv_perf_valid,
                                 " | FLOPs/cycle " + perf_field(rvv_perf_valid, compute_flops_per_cycle, 2, ""));
                } else {
                    report_row("hardware counters", "unavailable", "perf_event_open() is not permitted or not supported here");
                }
#else
                report_row("pack vs compute", "not available",
                           "per-tile timers are compiled out of this build; rerun with --instr for the split and the hardware counters");
#endif
                cout << "  +- " << (status == "PASS" ? "✅ PASS [result matches the naive scalar GEMM]"
                                                     : "⏭️ SKIP [correctness not checked: skip_ref_run in config.json]")
                     << " [cache blocking: mc=" << mc << " rows of A per block, nc=" << nc
                     << " columns of B per block, kc=" << kc << " depth of each block]" << endl;
            }

            free(Ac);
            free(Bc);
        }

        // Print Top 3 for this problem
        cout << "\r                                                                               \r";
        sort(variant_results.begin(), variant_results.end(), [](const VariantResult& a, const VariantResult& b) {
            if (a.status == "PASS" && b.status != "PASS") return true;
            if (a.status != "PASS" && b.status == "PASS") return false;
            return a.ker_gflops > b.ker_gflops;
        });

        string problem_str = label + " (" + dim_str + ")";
        stringstream ss;
        ss << "▶ " << left << setw(22) << problem_str << " | ";
        if (!variant_results.empty()) {
            auto& vr = variant_results[0];

            string short_name = short_variant_name(vr.name);

            ss << "best variant " << short_name << " [" << vr.mr << " rows x " << vr.nr
               << " columns of C per tile] | "
               << num(vr.e2e_gops, 2) << " GOPs by mean time [whole GEMM call, packing included"
#if RVV_INSTRUMENTED
               << "; DEBUG build, timers inflate this by 11-17%"
#endif
               << "] | " << num(vr.ker_gflops_median, 2) << " GOPs median rep | " << vr.n_reps << " reps | "
#if RVV_INSTRUMENTED
               << num(vr.rvv_bw, 1) << " GB/s [bytes packed / pack time] vector pack | "
               << num(vr.scl_bw, 1) << " GB/s scalar pack | compute IPC " << num(vr.compute_ipc, 2)
               << " [instructions per cycle] | "
#endif
               ;
            if (vr.status == "PASS") {
                ss << "✅ C[" << vr.sample_i << "," << vr.sample_j << "]: " << fixed << setprecision(2)
                   << vr.sample_ref << " == " << vr.sample_rvv << " [sample entry vs the naive scalar GEMM]";
            } else if (vr.status == "SKIP") {
                ss << "⏭️ SKIP [correctness not checked]";
            } else {
                ss << "❌ " << vr.status;
            }
            ss << " (cache blocking mc:" << vr.mc << " nc:" << vr.nc << " kc:" << vr.kc << ")";
        }
        string output_line = ss.str();
        cout << output_line << endl;
        
        if (summary_log.is_open()) {
            summary_log << output_line << "\n";
        }

        free(A);
        free(B);
        free(C);
        free(Cg);
    }

    perf_gemm_close(&perf);
    return 0;
}
