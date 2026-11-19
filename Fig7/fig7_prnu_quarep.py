#!/usr/bin/env python
'''fig 7: simulated photo-response non-uniformity (prnu) and its flat-field correction.

same style as fig 2 (183 mm wide, 7 pt sans-serif, lowercase bold panel letters, viridis/grey accents),
adapted to the quarep-limi checklists (schmied et al., nat methods 2024, doi 10.1038/s41592-023-01987-9).

usage:
    python fig7_prnu_quarep.py -o fig7_out
    python fig7_prnu_quarep.py -o fig7_out --frames 1000      # quick preview of panel e

simulation (all random draws seeded, every parameter is written to fig7_provenance.json):
    true gain map g      1 + N(0, PRNU_SIGMA) per pixel, with column-amplifier structure: a random half of the
                         columns in each sensor half at COLUMN_FACTOR, and the middle row at ROW_FACTOR
    estimated gain ghat  mean of N_FLAT flat-field frames at FLAT_E electrons, offset subtracted, normalised to mean 1
    frame                offset + read noise + poisson(g * scene), in adu (camera gain 1 adu per electron)
    correction           corrected = (raw - offset) / ghat + offset
panel e: N_FRAMES frames of a uniform scene at each level in LEVELS_E, same g and ghat; per-pixel mean and variance
over the frames, raw and corrected, against the poisson + read-noise model var = (mean - offset) + read_noise^2.
panel f: column means over the top sensor half of a and c.
panel g: at TEST_LEVEL_E, the fraction of frames in which each pixel lies outside +-Z_CRIT s.d. of the poisson +
read-noise model of its expected value (a 5 % test), against the pixel's true gain.
'''
import argparse
import json
import platform
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tifffile
from matplotlib.lines import Line2D

# ---- simulation settings ----------------------------------------------------------------------------
N_PX = 256                      # simulated sensor region, n x n pixels
OFFSET_ADU = 100.0              # camera offset (bias)
READ_NOISE_E = 1.0              # read noise, s.d. in electrons (= adu here)
PRNU_SIGMA = 0.05               # pixel-to-pixel gain s.d.; exaggerated for visibility (scmos is typically <1 %)
COLUMN_FACTOR = 0.95            # gain of the affected columns (column amplifiers)
ROW_FACTOR = 1.05               # gain of the middle row (boundary between the two sensor halves)
N_FLAT = 100                    # flat-field frames used to estimate the gain map
FLAT_E = 1000                   # mean signal of each flat-field frame (electrons)
BACKGROUND_E = 100              # uniform background of the scene in panels a-d (electrons)
SPOT_PEAK_E = 100               # peak of each gaussian spot above the background (electrons)
SPOT_SIGMA_PX = 50
SPOT_CENTRES = [(190, 70), (100, 190)]   # (row, col) of the spots
N_FRAMES = 10_000               # frames per illumination level in panel e
LEVELS_E = [0, 200, 400, 600, 800, 1000]  # uniform illumination levels in panel e (electrons)
SEED = 10
PROFILE_COLS = 64               # columns shown in the panel f profile
TEST_LEVEL_E = 1000             # uniform illumination level of the panel g noise-model test
Z_CRIT = 1.96                   # two-sided 5 % threshold of the per-pixel test

# ---- figure settings (match fig 2) ------------------------------------------------------------------
FIG_W_MM = 183
FIG_H_MM = 114
FONT_PT = 7
EXPORT_DPI = 600
DISPLAY_PERCENTILES = (0.1, 99.9)   # display range of a and c (shared), and the symmetric range of b and d
DIVERGING = 'PuOr_r'            # colour-blind-safe diverging map for the gain map and the difference
COLOUR_RAW = plt.cm.viridis(0.1)
COLOUR_CORR = plt.cm.viridis(0.6)


def make_gain_map(rng):
    '''true per-pixel gain: random prnu plus column and row structure.'''
    g = 1 + rng.normal(0, PRNU_SIGMA, (N_PX, N_PX))
    mid = N_PX // 2
    top_cols = rng.random(N_PX) < 0.5
    bottom_cols = rng.random(N_PX) < 0.5
    g[:mid, top_cols] *= COLUMN_FACTOR
    g[mid:, bottom_cols] *= COLUMN_FACTOR
    g[mid, :] *= ROW_FACTOR
    return g


def expose(rng, g, scene_e, n=None):
    '''camera frame(s) in adu: offset + read noise + poisson(g * scene).'''
    shape = scene_e.shape if n is None else (n,) + scene_e.shape
    return (OFFSET_ADU + rng.poisson(g * scene_e, size=shape)
            + rng.normal(0, READ_NOISE_E, shape)).astype(np.float64)


def estimate_gain(rng, g):
    '''flat-field estimate of the gain map from N_FLAT frames, normalised to mean 1.'''
    flats = expose(rng, g, np.full((N_PX, N_PX), float(FLAT_E)), n=N_FLAT)
    m = flats.mean(axis=0) - OFFSET_ADU
    return m / m.mean()


def correct(raw, ghat):
    return (raw - OFFSET_ADU) / ghat + OFFSET_ADU


def scene():
    rr, cc = np.mgrid[:N_PX, :N_PX]
    s = np.full((N_PX, N_PX), float(BACKGROUND_E))
    for r0, c0 in SPOT_CENTRES:
        s += SPOT_PEAK_E * np.exp(-((rr - r0) ** 2 + (cc - c0) ** 2) / SPOT_SIGMA_PX ** 2)
    return s


def mean_var(rng, g, ghat, level, n_frames, chunk=250, z_test=False):
    '''per-pixel mean and variance over n_frames uniform frames, raw and corrected (float64 sums).
    z_test: also the per-pixel fraction of frames outside +-Z_CRIT s.d. of a poisson + read-noise model
    centred on the pixel's expected value (what a noise-model-based test would flag as an event).'''
    flat = np.full((N_PX, N_PX), float(level))
    sums = {k: np.zeros((N_PX, N_PX)) for k in ('raw', 'corr')}
    sqs = {k: np.zeros((N_PX, N_PX)) for k in ('raw', 'corr')}
    hits = {k: np.zeros((N_PX, N_PX)) for k in ('raw', 'corr')}
    # expected signal above the offset of each pixel, and the variance a poisson model would assign it
    expected = {'raw': g * level, 'corr': g * level / ghat}
    sd_model = {k: np.sqrt(e + READ_NOISE_E ** 2) for k, e in expected.items()}
    # shift by the offset before summing so the variance does not lose precision
    for start in range(0, n_frames, chunk):
        raw = expose(rng, g, flat, n=min(chunk, n_frames - start))
        for k, x in (('raw', raw), ('corr', correct(raw, ghat))):
            x = x - OFFSET_ADU
            sums[k] += x.sum(axis=0)
            sqs[k] += (x ** 2).sum(axis=0)
            if z_test:
                hits[k] += (np.abs(x - expected[k]) > Z_CRIT * sd_model[k]).sum(axis=0)
    out = {}
    for k in sums:
        mean = sums[k] / n_frames
        var = (sqs[k] - n_frames * mean ** 2) / (n_frames - 1)
        out[k] = (mean + OFFSET_ADU, var)
        if z_test:
            out[k + '_fpr'] = hits[k] / n_frames
    return out


def image_panel(fig, rect_mm, img, cmap, vmin, vmax, title, cbar_label, ticks=None, fmt='{:.0f}'):
    '''image at native pixels with a horizontal calibration bar underneath; rect in mm from the top left.'''
    x, y, w, h = rect_mm
    ax = fig.add_axes(mm_rect(x, y, w, h))
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='none')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
    ax.set_title(title, fontsize=FONT_PT, pad=3)
    cax = fig.add_axes(mm_rect(x, y + h + 1.5, w, 1.6))
    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_ticks(ticks if ticks is not None else [vmin, vmax])
    cbar.set_ticklabels([fmt.format(t).replace('-', '\u2212') for t in cbar.get_ticks()])
    cbar.ax.tick_params(labelsize=FONT_PT - 1, length=1.5, width=0.4, pad=1)
    cbar.outline.set_linewidth(0.4)
    cbar.set_label(cbar_label, fontsize=FONT_PT - 1, labelpad=1)
    return ax


def dispersion_hist(ax, st):
    '''histogram of per-pixel variance divided by the poisson + read-noise model, uncorrected and corrected.'''
    bins = np.linspace(0.8, 1.2, 81)
    handles = []
    for key, colour, label in (('raw', COLOUR_RAW, 'Uncorrected'), ('corr', COLOUR_CORR, 'PRNU corrected')):
        m, v = st[key]
        ratio = (v / (m - OFFSET_ADU + READ_NOISE_E ** 2)).ravel()
        ax.hist(ratio, bins=bins, histtype='stepfilled', color=colour, alpha=0.3, linewidth=0)
        ax.hist(ratio, bins=bins, histtype='step', color=colour, linewidth=0.7)
        handles.append(mpl.patches.Patch(facecolor=mpl.colors.to_rgba(colour, 0.3), edgecolor=colour, linewidth=0.7,
                                         label=f'{label} (s.d. {ratio.std():.3f})'))
        print(f'e: {key} variance/model {ratio.mean():.3f} +- {ratio.std():.3f}, '
              f'{(ratio < bins[0]).mean() * 100 + (ratio > bins[-1]).mean() * 100:.2f} % of pixels outside the plotted range')
    ax.axvline(1, ymax=0.68, color='black', ls='--', lw=0.7)      # stops below the legend
    handles.append(Line2D([], [], color='black', ls='--', lw=0.7, label='Poisson + read noise'))
    ax.set_xlim(bins[0], bins[-1])
    ax.set_xlabel(f'Per-pixel variance / model ({TEST_LEVEL_E} e$^-$)')
    ax.set_ylabel('Pixels')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 3), useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(FONT_PT - 1)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=FONT_PT - 1, handlelength=1.6,
              borderaxespad=0.2)


def noise_test_panel(ax, g, st):
    '''percentage of frames a 5 % poisson + read-noise test flags in each pixel, against the pixel's true gain.'''
    show = np.random.default_rng(1).choice(N_PX * N_PX, size=4000, replace=False)
    edges = np.percentile(g, np.linspace(1, 99, 13))
    centres = (edges[:-1] + edges[1:]) / 2
    which = np.digitize(g.ravel(), edges) - 1
    binned = {}
    for key, colour, z in (('corr', COLOUR_CORR, 2), ('raw', COLOUR_RAW, 3)):
        fpr = st[key + '_fpr'].ravel() * 100
        ax.scatter(g.ravel()[show], fpr[show], s=0.6, color=colour, alpha=0.3, linewidths=0, rasterized=True,
                   zorder=z)
        # mean of each gain bin (equal-count bins), so the trend is readable through the scatter
        binned[key] = [fpr[which == i].mean() for i in range(len(centres))]
        ax.plot(centres, binned[key], '-o', color=colour, lw=0.9, ms=2.2, mec='white', mew=0.3, zorder=5)
        print(f'g: {key} flagged {fpr.mean():.2f} % (pixel s.d. {fpr.std():.2f} %), '
              f'bin means {binned[key][0]:.2f} % at gain {centres[0]:.3f} to {binned[key][-1]:.2f} % at {centres[-1]:.3f}')
    ax.axhline(5, color='black', ls='--', lw=0.7, zorder=4)
    ax.set_xlabel('True pixel gain')
    ax.set_ylabel(f'Frames outside \u00b1{Z_CRIT} s.d. (%)')
    ax.spines[['top', 'right']].set_visible(False)
    return centres, binned


def mm_rect(x, y, w, h):
    '''[left, bottom, width, height] in figure fractions from a rectangle in mm measured from the top left.'''
    return [x / FIG_W_MM, 1 - (y + h) / FIG_H_MM, w / FIG_W_MM, h / FIG_H_MM]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-o', '--out', type=Path, default=Path('fig7_out'), help='output folder')
    p.add_argument('--frames', type=int, default=N_FRAMES, help='frames per illumination level in panel e')
    p.add_argument('--formats', nargs='+', default=['pdf', 'tif', 'png'])
    p.add_argument('--e-style', choices=['scatter', 'hist'], default='scatter',
                   help='panel e as the mean-variance scatter, or as a histogram of variance / model at TEST_LEVEL_E')
    p.add_argument('--show', action='store_true', help='open the figure in a window')
    args = p.parse_args()

    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': FONT_PT, 'axes.labelsize': FONT_PT,
        'xtick.labelsize': FONT_PT - 1, 'ytick.labelsize': FONT_PT - 1,
        'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',     # keep text editable and as vector
    })

    rng = np.random.default_rng(SEED)
    g = make_gain_map(rng)
    ghat = estimate_gain(rng, g)
    raw = expose(rng, g, scene())
    corr = correct(raw, ghat)
    diff = raw - corr
    print(f'gain map: true s.d. {g.std():.4f}, estimate error s.d. {(ghat - g / g.mean()).std():.4f}')
    rel = g / ghat      # what the correction leaves of the gain pattern (flat-field correction is relative)
    print(f'pixel gain error: {(g / g.mean()).std() * 100:.2f} % uncorrected, {(rel / rel.mean()).std() * 100:.2f} % corrected')

    # panel f: column means over the top sensor half (one column-amplifier pattern), first PROFILE_COLS columns
    profiles = {name: img[:N_PX // 2, :PROFILE_COLS].mean(axis=0)
                for name, img in (('Uncorrected', raw), ('PRNU corrected', corr))}
    for name, prof in profiles.items():
        print(f'{name}: column-mean s.d. after removing a linear trend '
              f'{np.std(prof - np.polyval(np.polyfit(np.arange(PROFILE_COLS), prof, 1), np.arange(PROFILE_COLS))):.2f} adu')

    # panel e statistics
    stats = {}
    for level in LEVELS_E:
        stats[level] = mean_var(rng, g, ghat, level, args.frames, z_test=level == TEST_LEVEL_E)
        r, c = stats[level]['raw'], stats[level]['corr']
        print(f'level {level:5d} e-: raw var/(mean-offset+rn^2) {np.mean(r[1] / (r[0] - OFFSET_ADU + READ_NOISE_E ** 2)):.3f} '
              f'(s.d. {np.std(r[1] / (r[0] - OFFSET_ADU + READ_NOISE_E ** 2)):.3f}), '
              f'corrected {np.mean(c[1] / (c[0] - OFFSET_ADU + READ_NOISE_E ** 2)):.3f} '
              f'(s.d. {np.std(c[1] / (c[0] - OFFSET_ADU + READ_NOISE_E ** 2)):.3f})')

    fig = plt.figure(figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4))

    def letter(x_mm, y_mm, t):
        fig.text(x_mm / FIG_W_MM, 1 - y_mm / FIG_H_MM, t, fontsize=FONT_PT + 1, fontweight='bold', va='top', ha='left')

    # row 1, panels a-d: the correction pipeline left to right, each image with its calibration bar underneath
    P, Y1 = 38, 8                           # image size and top edge (mm)
    xs = (7, 52, 97, 142)
    lo, hi = np.percentile(np.concatenate([raw.ravel(), corr.ravel()]), DISPLAY_PERCENTILES)
    g_lim = np.percentile(np.abs(ghat - 1), DISPLAY_PERCENTILES[1])
    d_lim = np.percentile(np.abs(diff), DISPLAY_PERCENTILES[1])
    panels = [
        ('a', raw, 'gray', lo, hi, 'Uncorrected', 'Intensity (ADU)', None, '{:.0f}'),
        ('b', ghat, DIVERGING, 1 - g_lim, 1 + g_lim, 'Estimated gain map', 'Relative gain',
         [1 - g_lim, 1, 1 + g_lim], '{:.2f}'),
        ('c', corr, 'gray', lo, hi, 'PRNU corrected', 'Intensity (ADU)', None, '{:.0f}'),
        ('d', diff, DIVERGING, -d_lim, d_lim, 'Uncorrected \u2212 corrected', 'Difference (ADU)',
         [-d_lim, 0, d_lim], '{:.0f}'),
    ]
    for x, (t, img, cmap, vmin, vmax, title, label, ticks, fmt) in zip(xs, panels):
        image_panel(fig, (x, Y1, P, P), img, cmap, vmin, vmax, title, label, ticks, fmt)
        letter(x - 5.5, Y1 - 4.5, t)

    # row 2, panels e-g. the legend in e explains the colours for f and g as well
    Y2, H2 = 66, 38
    ax = fig.add_axes(mm_rect(17, Y2, 45, H2))
    show = np.random.default_rng(0).choice(N_PX * N_PX, size=min(4000, N_PX * N_PX), replace=False)
    if args.e_style == 'hist':
        dispersion_hist(ax, stats[TEST_LEVEL_E])
    else:
        for level in LEVELS_E:
            for key, colour, z in (('raw', COLOUR_RAW, 2), ('corr', COLOUR_CORR, 3)):
                m, v = stats[level][key]
                ax.scatter(m.ravel()[show], v.ravel()[show], s=0.5, color=colour, alpha=0.35, linewidths=0,
                           rasterized=True, zorder=z)
        m_line = np.array([OFFSET_ADU, OFFSET_ADU + max(LEVELS_E) * 1.15])
        ax.plot(m_line, m_line - OFFSET_ADU + READ_NOISE_E ** 2, color='black', ls='--', lw=0.7, zorder=4)
        ax.set_xlabel('Per-pixel mean (ADU)')
        ax.set_ylabel('Per-pixel variance (ADU$^2$)')
        ax.set_xlim(OFFSET_ADU - 40, None)
        ax.set_ylim(-30, None)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(handles=[
            Line2D([], [], ls='', marker='o', ms=3, color=COLOUR_RAW, label='Uncorrected'),
            Line2D([], [], ls='', marker='o', ms=3, color=COLOUR_CORR, label='PRNU corrected'),
            Line2D([], [], color='black', ls='--', lw=0.7, label='Poisson + read noise'),
        ], loc='upper left', frameon=False, fontsize=FONT_PT - 1, handlelength=2.2, borderaxespad=0.2)
    letter(4, Y2 - 4.5, 'e')

    # f: column means of the top sensor half of a and c, showing the column pattern that the correction removes
    ax_f = fig.add_axes(mm_rect(78, Y2, 42, H2))
    cols = np.arange(PROFILE_COLS)
    for colour, label in ((COLOUR_RAW, 'Uncorrected'), (COLOUR_CORR, 'PRNU corrected')):
        ax_f.plot(cols, profiles[label], '-', color=colour, lw=0.8)
    ax_f.set_xlim(-1, PROFILE_COLS)
    ax_f.set_xlabel('Column (px)')
    ax_f.set_ylabel(f'Mean of rows 0\u2013{N_PX // 2 - 1} (ADU)')
    ax_f.spines[['top', 'right']].set_visible(False)
    letter(66, Y2 - 4.5, 'f')

    # g: a 5 % noise-model test per pixel at TEST_LEVEL_E, against the pixel's true gain
    ax_g = fig.add_axes(mm_rect(137, Y2, 43, H2))
    centres, binned = noise_test_panel(ax_g, g, stats[TEST_LEVEL_E])
    letter(124, Y2 - 4.5, 'g')

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        kw = {'pil_kwargs': {'compression': 'tiff_lzw'}} if fmt in ('tif', 'tiff') else {}
        fig.savefig(out / f'fig7_prnu.{fmt}', dpi=EXPORT_DPI, **kw)

    # exact arrays shown in a-d (float32 tiff), and panel e per-pixel statistics
    panel_dir = out / 'fig7_panel_data'
    panel_dir.mkdir(exist_ok=True)
    for name, img in (('a_uncorrected', raw), ('b_estimated_gain', ghat), ('c_corrected', corr),
                      ('d_difference', diff), ('true_gain', g)):
        tifffile.imwrite(panel_dir / f'{name}.tif', img.astype(np.float32))
    np.savez_compressed(out / 'fig7_e_mean_var.npz', levels_e=np.array(LEVELS_E),
                        **{f'{k}_{q}_{lv}': stats[lv][k][i] for lv in LEVELS_E for k in ('raw', 'corr')
                           for i, q in enumerate(('mean', 'var'))})
    with open(out / 'fig7_source_data.csv', 'w') as fh:
        fh.write('level_e,image,mean_of_means_adu,mean_of_variances_adu2,sd_of_variances_adu2,model_variance_adu2\n')
        for lv in LEVELS_E:
            for k in ('raw', 'corr'):
                m, v = stats[lv][k]
                fh.write(f'{lv},{"uncorrected" if k == "raw" else "corrected"},{m.mean():.4f},{v.mean():.4f},'
                         f'{v.std():.4f},{m.mean() - OFFSET_ADU + READ_NOISE_E ** 2:.4f}\n')

    with open(out / 'fig7_f_profile.csv', 'w') as fh:
        fh.write('column,uncorrected_adu,corrected_adu\n')
        for c in range(PROFILE_COLS):
            fh.write(f'{c},{profiles["Uncorrected"][c]:.4f},{profiles["PRNU corrected"][c]:.4f}\n')
    st = stats[TEST_LEVEL_E]
    np.savez_compressed(out / 'fig7_g_noise_test.npz', true_gain=g, flagged_uncorrected=st['raw_fpr'],
                        flagged_corrected=st['corr_fpr'])
    with open(out / 'fig7_g_binned.csv', 'w') as fh:
        fh.write('gain_bin_centre,uncorrected_flagged_pct,corrected_flagged_pct\n')
        for c, r, k in zip(centres, binned['raw'], binned['corr']):
            fh.write(f'{c:.4f},{r:.4f},{k:.4f}\n')

    provenance = {
        'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
        'matplotlib': mpl.__version__, 'tifffile': tifffile.__version__,
        'seed': SEED, 'n_px': N_PX, 'offset_adu': OFFSET_ADU, 'camera_gain_adu_per_e': 1.0,
        'read_noise_e': READ_NOISE_E, 'prnu_sigma': PRNU_SIGMA, 'column_factor': COLUMN_FACTOR,
        'row_factor': ROW_FACTOR, 'n_flat': N_FLAT, 'flat_e': FLAT_E, 'background_e': BACKGROUND_E,
        'spot_peak_e': SPOT_PEAK_E, 'spot_sigma_px': SPOT_SIGMA_PX, 'spot_centres_rc': SPOT_CENTRES,
        'n_frames_e': args.frames, 'levels_e': LEVELS_E, 'test_level_e_g': TEST_LEVEL_E, 'z_crit_g': Z_CRIT,
        'profile_rows_f': [0, N_PX // 2 - 1], 'profile_cols_f': [0, PROFILE_COLS - 1],
        'display': {'a_c_range_adu': [lo, hi], 'b_range': [1 - g_lim, 1 + g_lim], 'd_range_adu': [-d_lim, d_lim],
                    'percentiles': list(DISPLAY_PERCENTILES), 'pixels_plotted_per_level_e': int(len(show))},
    }
    (out / 'fig7_provenance.json').write_text(json.dumps(provenance, indent=2))
    print(f'saved to {out.resolve()}')
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
