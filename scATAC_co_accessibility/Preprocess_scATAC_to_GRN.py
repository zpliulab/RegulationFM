import sys
import os
import pandas as pd
gimmemotifs_env_path = "/home/wcy/miniconda3/envs/celloracle_env/lib/python3.10/site-packages/"
sys.path.append(gimmemotifs_env_path)
import subprocess
import numpy as np
import scATAC_co_accessibility.motif_analysis as ma
from config import base_GRN_Config

def tif2dataframe(tfi):
    tfi.reset_filtering()
    tfi.filter_motifs_by_score(threshold=10)
    tfi.make_TFinfo_dataframe_and_dictionary(verbose=True)
    df = tfi.to_dataframe()
    df = df.groupby('gene_short_name', as_index=False).sum()
    df = df.drop(columns=['peak_id'])
    df.set_index(df.columns[0], inplace=True)
    df = df.reset_index().melt(id_vars='gene_short_name', var_name='TF', value_name='weight')
    df.columns = ['Gene', 'TF', 'weight']
    df = df[df['weight'] != 0]
    return df


def scan_motif(filename_peak_coaccess, ref_genome, fpr=0.01):
    genome_installation = ma.is_genome_installed(ref_genome=ref_genome,
                                                 genomes_dir=None)
    print(ref_genome, "installation: ", genome_installation)
    if not genome_installation:
        import genomepy
        genomepy.install_genome(name=ref_genome, provider="UCSC", genomes_dir=None)
    else:
        print(ref_genome, "is installed.")

    peaks = pd.read_csv(filename_peak_coaccess, index_col=0)
    peaks = ma.check_peak_format(peaks, ref_genome, genomes_dir=None)

    # Instantiate TFinfo object
    tfi = ma.TFinfo(peak_data_frame=peaks,
                    ref_genome=ref_genome,
                    genomes_dir=None)

    # Scan motifs. !!CAUTION!! This step may take several hours if you have many peaks!
    tfi.scan(fpr=fpr,
             motifs=None,  # If you enter None, default motifs will be loaded.
             verbose=True,
             n_cpus=1,
             batch_size=30000)
    return tfi


def trans_coa_gene_name(filename_cicero_con, filename_peaknames, filename_peak_coaccess, ref_genome, thre_coa, Only_promoter):
    peaks = pd.read_csv(filename_peaknames, index_col=0)
    peaks = peaks.x.values
    peaks = np.vectorize(lambda x: x.replace(':', '_').replace('-', '_') if isinstance(x, str) else x)(peaks)
    cicero_connections = pd.read_csv(filename_cicero_con, index_col=0)
    cicero_connections = cicero_connections.applymap(
        lambda x: x.replace(':', '_').replace('-', '_') if isinstance(x, str) else x)

    tss_annotated = ma.get_tss_info(peak_str_list=peaks,
                                    ref_genome=ref_genome,
                                    custom_tss_file_path=os.path.join('scATAC_co_accessibility/motif_analysis',
                                                                      "tss_ref_data",
                                                                      f"{ref_genome}_tss_info.bed"),
                                    Only_promoter=Only_promoter)
    integrated = ma.integrate_tss_peak_with_cicero(tss_peak=tss_annotated,
                                                   cicero_connections=cicero_connections)
    peak = integrated[integrated.coaccess >= thre_coa]
    peak = peak[["peak_id", "gene_short_name", "coaccess"]].reset_index(drop=True)
    if Only_promoter:
        filename_peak_coaccess = filename_peak_coaccess.replace('genename', 'Prom_genename')
    peak.to_csv(filename_peak_coaccess)


def run_constructed_base_GRN(ALL_data_file_path, cell_type, run_R_code=False):
    args = base_GRN_Config()
    ref_genome = args.ref_genome
    species = args.species

    data_folder = os.path.join(ALL_data_file_path, 'cicero_output', cell_type)
    filename_base_GRN = os.path.join(data_folder, args.out_path)
    filename_peaknames = os.path.join(data_folder, "all_peaks.csv")
    filename_cicero_con = os.path.join(data_folder, "cicero_connections.csv")
    filename_peak_coaccess = os.path.join(data_folder, "peak_coa_trans_genename.csv")

    # Step1 - run cicero on atac
    if run_R_code:
        Rcommnd = f'Rscript ./scATAC_co_accessibility/step1_preprocess_atac2peak.R {ALL_data_file_path} {species} {cell_type}'
        env = {'PATH': '/home/wcy/miniconda3/envs/Multiomics/bin'}
        subprocess.run(Rcommnd, env=env, shell=True)

    if not os.path.exists(filename_cicero_con):
        return None

    # Step2 - preprocess_peak_data
    # 2.1 Annotate and extract the core network of promoters and enhancers
    trans_coa_gene_name(filename_cicero_con,
                        filename_peaknames,
                        filename_peak_coaccess,
                        ref_genome, thre_coa=args.prom_thre_coa,
                        Only_promoter=True)

    # 2.2 Annotate and extract the network of all DNA fragments
    trans_coa_gene_name(filename_cicero_con,
                        filename_peaknames,
                        filename_peak_coaccess,
                        ref_genome, thre_coa=args.all_thre_coa,
                        Only_promoter=False)

    # Step3 - motif scan
    tfi_prom = scan_motif(filename_peak_coaccess.replace('genename', 'Prom_genename'), ref_genome)
    tfi = scan_motif(filename_peak_coaccess, ref_genome)

    # Step4 - trans and combine
    Link_prom = tif2dataframe(tfi_prom)
    Link_All = tif2dataframe(tfi)
    Link_All['combine'] = Link_All['Gene'] + '_' + Link_All['TF']
    Link_prom['combine'] = Link_prom['Gene'] + '_' + Link_prom['TF']
    Link_prom['type'] = 'promote'
    Link_All['type'] = 'All'
    Link_combined = pd.concat([Link_prom, Link_All])
    Link_combined = Link_combined.drop_duplicates(subset='combine', keep='first')
    print('INFO:: Link (annotate promoter/enhancer):', Link_prom.shape[0])
    print('INFO:: Link (annotate all gene structures):', Link_All.shape[0])
    print('INFO:: The num of Link after combined is:', Link_combined.shape[0])
    # save pandas
    Link_combined.to_parquet(filename_base_GRN)
    print(f'INFO:: base_GRN constructed successfully, and Dataframe saved at {filename_base_GRN}')
    return Link_combined

def from_ATAC_constructed_base_GRN(ALL_data_file_path, cell_type, run_R_code=False, rerun=False):
    args = base_GRN_Config()
    filename_base_GRN = os.path.join(ALL_data_file_path, 'cicero_output', cell_type, args.out_path)
    if not os.path.exists(filename_base_GRN):
        print(f'INFO:: base_GRN does not exist. Recalculating!')
        rerun = True

    if rerun:
        Link_combined = run_constructed_base_GRN(ALL_data_file_path,
                                                 cell_type,
                                                 run_R_code=run_R_code)
    else:
        filename_base_GRN = os.path.join(ALL_data_file_path, 'cicero_output', cell_type, args.out_path)
        if os.path.exists(filename_base_GRN):
            Link_combined = pd.read_parquet(filename_base_GRN)
        else:
            Link_combined = None
            print(f'INFO:: {filename_base_GRN}  does not exist. Using public base_GRN!')
    Link_human = pd.read_parquet('/home/wcy/RegulationGPT/Data/promoter_base_GRN/hg38_TFinfo_dataframe_gimmemotifsv5_fpr1_threshold_10_20210630.parquet')
    Link_human = Link_human.drop(columns=['peak_id'])
    Link_human.set_index(Link_human.columns[0], inplace=True)
    Link_human = Link_human.reset_index().melt(id_vars='gene_short_name', var_name='TF', value_name='weight')
    Link_human.columns = ['Gene', 'TF', 'weight']
    Link_human = Link_human[Link_human['weight'] != 0]
    Link_human['combine'] = Link_human['Gene'] + '_' + Link_human['TF']
    Link_human['type'] = 'public'
    Link_combined = pd.concat([Link_combined, Link_human])
    Link_combined = Link_combined.drop_duplicates(subset=['combine'], keep='first')

    return Link_combined


if __name__ == '__main__':
   # import motif_analysis as ma
    #from ..config import base_GRN_Config
    # main_Config
    import argparse
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--data_folder', type=str, default='/home/wcy/RegulationGPT/Data/PBMC_Healthy_Donor_3k', help='Your username')
    args = parser.parse_args()
    data_folder = args.data_folder
#    data_folder = '/home/wcy/RegulationGPT/Data/PBMC_Healthy_Donor_3k'
    file_names = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith("label.csv")]
    cell_type = pd.read_csv(file_names[0])
    unique_cell_type = np.unique(cell_type.iloc[:, 1])
    print(f'INFO: {data_folder}')
    for select_cell_type in unique_cell_type:
        if len(os.listdir(os.path.join(data_folder, 'cicero_output', select_cell_type))) != 0:
            base_GRN_link = from_ATAC_constructed_base_GRN(data_folder, select_cell_type, run_R_code=False, rerun=True)
            print(f'INFO: {select_cell_type} is pre-process OK!')

    # path = '/home/wcy/RegulationGPT/Data/PBMC/cicero_output/base_GRN_dataframe.parquet'
    # B1 = pd.read_parquet(path)
    # result = data.groupby(data.columns.tolist(), as_index=False).size()
else:
    import scATAC_co_accessibility.motif_analysis as ma
    from config import base_GRN_Config
