#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# feature parsing
# =========================================================

def split_feature_parts(feature: str):
    """
    ASTRO feature examples:
      Xkr4__protein_coding__intron
      Lypla1__protein_coding__exon
      ENSMUSG00000121478__lncRNA__exon
      Mus_musculus_tRNA-Leu-CAA-3-1__tRNA
      Rptor__protein_coding__exon---ENSMUSG00000121577__protein_coding__exon
    """
    return [x.strip() for x in str(feature).split('---') if x.strip()]


def first_token_before_double_underscore(part: str):
    return str(part).split('__')[0].strip()


def parse_feature_class(part: str):
    s = str(part).lower()
    if '__exon' in s:
        return 'exon'
    elif '__intron' in s:
        return 'intron'
    else:
        return 'other'


def parse_biotype(part: str):
    toks = str(part).split('__')
    if len(toks) >= 2:
        return toks[1]
    return 'other'


# =========================================================
# species inference
# =========================================================

human_regexes = [
    re.compile(r'ENSG\d+', re.I),
    re.compile(r'ENST\d+', re.I),
    re.compile(r'Homo_sapiens', re.I),
    re.compile(r'\bhg38\b', re.I),
    re.compile(r'GRCh38', re.I),
]

mouse_regexes = [
    re.compile(r'ENSMUSG\d+', re.I),
    re.compile(r'ENSMUST\d+', re.I),
    re.compile(r'Mus_musculus', re.I),
    re.compile(r'\bmm10\b', re.I),
    re.compile(r'\bmm39\b', re.I),
    re.compile(r'GRCm38', re.I),
    re.compile(r'GRCm39', re.I),
]


def looks_like_human_symbol(tok: str):
    """
    # Common human gene-symbol format:
      TP53, ACTB, GAPDH, MALAT1, HLA-A, RPLP0, MT-ND1
    """
    tok = str(tok).strip()
    if tok == "":
        return False
    if re.fullmatch(r'[A-Z0-9\-.]+', tok):
        return re.search(r'[A-Z]', tok) is not None
    return False


def looks_like_mouse_symbol(tok: str):
    """
    # Common mouse gene-symbol format:
      Xkr4, Lypla1, Rgs20, Oprk1, Tcea1, Atp6v1h, Gm1992, 4732440D04Rik
    """
    tok = str(tok).strip()
    if tok == "":
        return False

    if re.fullmatch(r'Gm\d+', tok):
        return True

    if re.fullmatch(r'[A-Z][a-z0-9\-]+', tok):
        return True

    if re.fullmatch(r'\d+[A-Za-z]*Rik', tok):
        return True

    return False


def infer_species_from_part(part: str):
    hit_h = any(r.search(part) for r in human_regexes)
    hit_m = any(r.search(part) for r in mouse_regexes)

    tok = first_token_before_double_underscore(part)

    if hit_h and not hit_m:
        return 'human'
    if hit_m and not hit_h:
        return 'mouse'
    if hit_h and hit_m:
        return 'ambiguous'

    is_h_symbol = looks_like_human_symbol(tok)
    is_m_symbol = looks_like_mouse_symbol(tok)

    if is_h_symbol and not is_m_symbol:
        return 'human'
    if is_m_symbol and not is_h_symbol:
        return 'mouse'
    if is_h_symbol and is_m_symbol:
        return 'ambiguous'

    return 'unknown'


def infer_species(feature: str):
    parts = split_feature_parts(feature)
    calls = [infer_species_from_part(p) for p in parts]

    has_h = 'human' in calls
    has_m = 'mouse' in calls

    if has_h and not has_m:
        return 'human'
    if has_m and not has_h:
        return 'mouse'
    if has_h and has_m:
        return 'ambiguous_feature'

    return 'unknown_feature'


# =========================================================
# feature filter: final version only keeps protein_coding exon
# =========================================================

def keep_feature(feature: str):
    parts = split_feature_parts(feature)
    classes = [parse_feature_class(p) for p in parts]
    biotypes = [parse_biotype(p) for p in parts]

    return any((c == 'exon' and b == 'protein_coding') for c, b in zip(classes, biotypes))


# =========================================================
# barcode classification
# =========================================================

def classify_barcode(h, m, min_total=200, dom_frac=0.90, mixed_minor_frac=0.15, min_minor_count=30):
    total = h + m
    if total < min_total:
        return 'low_count'

    hf = h / total if total > 0 else 0.0
    mf = m / total if total > 0 else 0.0
    minor_frac = min(hf, mf)
    minor_count = min(h, m)

    if hf >= dom_frac:
        return 'human'
    if mf >= dom_frac:
        return 'mouse'
    if minor_frac >= mixed_minor_frac and minor_count >= min_minor_count:
        return 'mixed'
    return 'ambiguous'


def estimated_collision_rate(n_h, n_m, n_mix):
    """
    For an approximately 1:1 human-mouse mixture, cross-species doublets are expected to represent roughly half of all random two-cell collisions.
    The total collision rate is therefore estimated as approximately 2 * observed_mixed_fraction.
    """
    denom = n_h + n_m + n_mix
    if denom == 0:
        return np.nan
    return 2.0 * (n_mix / denom)


# =========================================================
# plotting helpers
# =========================================================

def make_barnyard_plot_linear(df, sample, out_pdf, min_total, dom_frac, mixed_minor_frac, min_minor_count):
    plot_df = df[df['total'] >= min_total].copy()

    colors = {
        'human': '#b5651d',
        'mouse': '#5e3c99',
        'mixed': '#bdbdbd',
        'ambiguous': '#d9d9d9',
    }

    fig, ax = plt.subplots(figsize=(7.8, 7.2))

    for cls in ['ambiguous', 'mixed', 'human', 'mouse']:
        sub = plot_df[plot_df['call'] == cls]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub['human'],
            sub['mouse'],
            s=10,
            c=colors[cls],
            alpha=0.75,
            linewidths=0,
            rasterized=True,
            label=f'{cls.capitalize()} ({len(sub)})'
        )

    xmax = float(plot_df['human'].max()) if len(plot_df) else 1.0
    ymax = float(plot_df['mouse'].max()) if len(plot_df) else 1.0
    xymax = max(xmax, ymax, 1.0)
    lim = xymax * 1.03

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    ax.plot([0, lim], [0, lim], ls='--', lw=1.0, c='gray')

    # Reference threshold for the minor-species transcript count
    ax.axvline(min_minor_count, ls=':', lw=1.0, c='gray')
    ax.axhline(min_minor_count, ls=':', lw=1.0, c='gray')

    ax.set_xlabel('Human transcripts / UMI per barcode', fontsize=15)
    ax.set_ylabel('Mouse transcripts / UMI per barcode', fontsize=15)
    ax.set_title(f'{sample} barnyard plot', fontsize=18)

    n_h = int((plot_df['call'] == 'human').sum())
    n_m = int((plot_df['call'] == 'mouse').sum())
    n_x = int((plot_df['call'] == 'mixed').sum())
    n_a = int((plot_df['call'] == 'ambiguous').sum())
    obs_mix = n_x / max((n_h + n_m + n_x), 1)
    coll = estimated_collision_rate(n_h, n_m, n_x)

    txt = (
        f'Plotted barcodes (total >= {min_total}): {len(plot_df)}\n'
        f'Human: {n_h}\n'
        f'Mouse: {n_m}\n'
        f'Mixed: {n_x}\n'
        f'Ambiguous: {n_a}\n'
        f'Observed mixed fraction: {obs_mix*100:.2f}%\n'
        f'Estimated collision rate: {coll*100:.2f}%\n'
        f'Pure cutoff: dominant fraction >= {dom_frac:.2f}\n'
        f'Mixed cutoff: minor fraction >= {mixed_minor_frac:.2f}\n'
        f'and minor count >= {min_minor_count}'
    )

    ax.text(
        0.03, 0.97, txt,
        transform=ax.transAxes,
        va='top', ha='left',
        fontsize=10.5,
        bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='lightgray', alpha=0.95)
    )

    ax.legend(frameon=False, loc='lower right', fontsize=11, markerscale=1.8)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_barnyard_plot_log(df, sample, out_pdf, min_total):
    plot_df = df[df['total'] >= min_total].copy()
    plot_df['human_plot'] = plot_df['human'] + 1
    plot_df['mouse_plot'] = plot_df['mouse'] + 1

    colors = {
        'human': '#b5651d',
        'mouse': '#5e3c99',
        'mixed': '#bdbdbd',
        'ambiguous': '#d9d9d9',
    }

    fig, ax = plt.subplots(figsize=(7.8, 7.2))

    for cls in ['ambiguous', 'mixed', 'human', 'mouse']:
        sub = plot_df[plot_df['call'] == cls]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub['human_plot'],
            sub['mouse_plot'],
            s=6,
            c=colors[cls],
            alpha=0.75,
            linewidths=0,
            rasterized=True,
            label=f'{cls.capitalize()} ({len(sub)})'
        )

    vmax = max(plot_df['human_plot'].max(), plot_df['mouse_plot'].max(), 10)
    vmax = 10 ** math.ceil(math.log10(vmax))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1, vmax)
    ax.set_ylim(1, vmax)
    ax.plot([1, vmax], [1, vmax], ls='--', lw=1.0, c='gray')

    ax.set_xlabel('Human transcripts / UMI per barcode (+1, log scale)', fontsize=14)
    ax.set_ylabel('Mouse transcripts / UMI per barcode (+1, log scale)', fontsize=14)
    ax.set_title(f'{sample} barnyard plot (log-scale supplement)', fontsize=17)

    ax.legend(frameon=False, loc='lower right', fontsize=11, markerscale=1.6)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_rank_plot(df, sample, out_pdf):
    tmp = df.sort_values('total', ascending=False).reset_index(drop=True)
    tmp['rank'] = np.arange(1, len(tmp) + 1)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(tmp['rank'], tmp['total'] + 1, lw=1.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Barcode rank', fontsize=14)
    ax.set_ylabel('Human + mouse transcripts / UMI', fontsize=14)
    ax.set_title(f'{sample} barcode rank', fontsize=17)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_fraction_hist(df, sample, out_pdf, min_total):
    sub = df[df['total'] >= min_total].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3))

    axes[0].hist(sub['human_frac'], bins=50)
    axes[0].set_title('Human fraction', fontsize=15)
    axes[0].set_xlabel('human / (human + mouse)', fontsize=13)
    axes[0].set_ylabel('Barcode count', fontsize=13)

    axes[1].hist(sub['mouse_frac'], bins=50)
    axes[1].set_title('Mouse fraction', fontsize=15)
    axes[1].set_xlabel('mouse / (human + mouse)', fontsize=13)
    axes[1].set_ylabel('Barcode count', fontsize=13)

    fig.suptitle(f'{sample} species fraction distribution', fontsize=18)
    plt.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    plt.close(fig)


# =========================================================
# main
# =========================================================

def main():
    ap = argparse.ArgumentParser(description='Final barnyard analysis from ASTRO expmat.tsv')
    ap.add_argument('--input', required=True, help='ASTRO expmat.tsv')
    ap.add_argument('--outdir', required=True, help='output directory')
    ap.add_argument('--sample', required=True, help='sample name')
    ap.add_argument('--chunksize', type=int, default=2000)
    ap.add_argument('--min-total', type=int, default=200)
    ap.add_argument('--dominant-frac', type=float, default=0.90)
    ap.add_argument('--mixed-minor-frac', type=float, default=0.15)
    ap.add_argument('--min-minor-count', type=int, default=30)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    header = pd.read_csv(args.input, sep='\t', nrows=0)
    cols = list(header.columns)
    if len(cols) < 2:
        raise ValueError('Input matrix malformed: fewer than 2 columns.')

    feature_col = cols[0]
    barcode_cols = cols[1:]

    human_sum = np.zeros(len(barcode_cols), dtype=np.float64)
    mouse_sum = np.zeros(len(barcode_cols), dtype=np.float64)

    total_rows = 0
    kept_rows = 0
    human_feature_rows = 0
    mouse_feature_rows = 0
    ambiguous_feature_rows = 0
    unknown_feature_rows = 0

    for chunk in pd.read_csv(args.input, sep='\t', chunksize=args.chunksize):
        total_rows += len(chunk)

        features = chunk.iloc[:, 0].astype(str)
        keep_mask = features.apply(keep_feature).values
        if keep_mask.sum() == 0:
            continue

        sub = chunk.loc[keep_mask].copy()
        kept_rows += len(sub)

        feats = sub.iloc[:, 0].astype(str).tolist()
        mat = sub.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy(dtype=np.float64)

        calls = np.array([infer_species(f) for f in feats], dtype=object)

        h_mask = (calls == 'human')
        m_mask = (calls == 'mouse')
        a_mask = (calls == 'ambiguous_feature')
        u_mask = (calls == 'unknown_feature')

        human_feature_rows += int(h_mask.sum())
        mouse_feature_rows += int(m_mask.sum())
        ambiguous_feature_rows += int(a_mask.sum())
        unknown_feature_rows += int(u_mask.sum())

        if h_mask.sum() > 0:
            human_sum += mat[h_mask, :].sum(axis=0)
        if m_mask.sum() > 0:
            mouse_sum += mat[m_mask, :].sum(axis=0)

    out = pd.DataFrame({
        'barcode': barcode_cols,
        'human': human_sum.astype(np.int64),
        'mouse': mouse_sum.astype(np.int64),
    })

    out['total'] = out['human'] + out['mouse']
    out['human_frac'] = np.where(out['total'] > 0, out['human'] / out['total'], 0.0)
    out['mouse_frac'] = np.where(out['total'] > 0, out['mouse'] / out['total'], 0.0)

    out['call'] = [
        classify_barcode(
            h, m,
            min_total=args.min_total,
            dom_frac=args.dominant_frac,
            mixed_minor_frac=args.mixed_minor_frac,
            min_minor_count=args.min_minor_count
        )
        for h, m in zip(out['human'], out['mouse'])
    ]

    plotted = out[out['total'] >= args.min_total].copy()
    n_h = int((plotted['call'] == 'human').sum())
    n_m = int((plotted['call'] == 'mouse').sum())
    n_x = int((plotted['call'] == 'mixed').sum())
    n_a = int((plotted['call'] == 'ambiguous').sum())
    obs_mix = n_x / max((n_h + n_m + n_x), 1)
    coll = estimated_collision_rate(n_h, n_m, n_x)

    calls_tsv = os.path.join(args.outdir, f'{args.sample}_species_calls.tsv')
    summary_tsv = os.path.join(args.outdir, f'{args.sample}_summary.tsv')
    main_pdf = os.path.join(args.outdir, f'{args.sample}_barnyard_publication_linear.pdf')
    supp_pdf = os.path.join(args.outdir, f'{args.sample}_barnyard_log_supplement.pdf')
    rank_pdf = os.path.join(args.outdir, f'{args.sample}_barcode_rank.pdf')
    frac_pdf = os.path.join(args.outdir, f'{args.sample}_species_fraction_hist.pdf')

    out.to_csv(calls_tsv, sep='\t', index=False)

    summary = pd.DataFrame({
        'metric': [
            'sample',
            'input',
            'feature_col',
            'feature_mode',
            'total_feature_rows',
            'kept_feature_rows',
            'human_feature_rows',
            'mouse_feature_rows',
            'ambiguous_feature_rows',
            'unknown_feature_rows',
            'barcode_count',
            f'barcode_count_total_ge_{args.min_total}',
            'human_barcodes',
            'mouse_barcodes',
            'mixed_barcodes',
            'ambiguous_barcodes',
            'observed_mixed_fraction',
            'estimated_collision_rate',
            'dominant_frac_cutoff',
            'mixed_minor_frac_cutoff',
            'min_minor_count_cutoff',
        ],
        'value': [
            args.sample,
            args.input,
            feature_col,
            'protein_coding_exon_only',
            total_rows,
            kept_rows,
            human_feature_rows,
            mouse_feature_rows,
            ambiguous_feature_rows,
            unknown_feature_rows,
            len(out),
            len(plotted),
            n_h,
            n_m,
            n_x,
            n_a,
            obs_mix,
            coll,
            args.dominant_frac,
            args.mixed_minor_frac,
            args.min_minor_count,
        ]
    })
    summary.to_csv(summary_tsv, sep='\t', index=False)

    make_barnyard_plot_linear(
        out, args.sample, main_pdf,
        args.min_total, args.dominant_frac,
        args.mixed_minor_frac, args.min_minor_count
    )

    make_barnyard_plot_log(
        out, args.sample, supp_pdf,
        args.min_total
    )

    make_rank_plot(out, args.sample, rank_pdf)
    make_fraction_hist(out, args.sample, frac_pdf, args.min_total)

    print('Done.')
    print('Calls   :', calls_tsv)
    print('Summary :', summary_tsv)
    print('Main    :', main_pdf)
    print('Supp    :', supp_pdf)
    print('Rank    :', rank_pdf)
    print('Hist    :', frac_pdf)


if __name__ == '__main__':
    main()
