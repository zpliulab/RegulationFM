
#rm(list=ls())
#print(pwd)
library(EnsDb.Hsapiens.v86)
.libPaths("/home/wcy/miniconda3/envs/publicR/lib/R/library")
library(cicero, lib.loc = "/home/wcy/miniconda3/envs/Multiomics/lib/R/library")
library(monocle3)
library(Seurat)
library(Signac,lib.loc = "/home/wcy/miniconda3/envs/DeepMAPS/lib/R/library")
library(rhdf5)
library(readr)
library(dplyr)

setwd('/home/wcy/RegulationGPT/scATAC_co_accessibility')


data_folder <- "/data/wcy_data/RegulationGPT_key/myocardial/new_preprocess/GT_IZ_P15/cicero_output/Myeloid"
args <- commandArgs(trailingOnly = TRUE)

data_folder <- args[1]
if (length(args) < 2){
  Species <- 'human'
}else{ Species <- args[2]}

if (length(args) < 3){
  select_celltype <- c()
}else{ select_celltype <- args[3]}

# Sub_function
### S1. 定义函数来检查文件夹中的必需文件
check_required_files_type <- function(search_directory) {
  files <- list.files(search_directory, pattern = "filtered_feature_bc_matrix\\.h5$", full.names = TRUE)
  files2 <- list.files(search_directory, pattern = "\\multiomics.h5$", full.names = TRUE)
  # 检查文件数量
  if (length(files) == 1) {
    return('10X_datasets')
  } else if (length(files2) == 1) {
    return('10X_visium')
  } else if (length(files) == 0) {
    return('other')
  } else {
    cat("找到多个文件，请确认:\n", paste(files, collapse = "\n"), "\n")
    break
  }
}

### S2. load data
sub_load_10X_data <- function(search_directory){

  file_names <- list.files(search_directory, pattern = "filtered_feature_bc_matrix\\.h5$", full.names = TRUE)

  alldata <- Read10X_h5(file_names)
  cellinfo <- read.csv(gsub('.h5','_label.csv',file_names), header=T)
  cellinfo <- as.data.frame(cellinfo)
  colnames(cellinfo) <- c("barcode", "celltype")
  rownames(cellinfo) <- cellinfo[,1]
  
  for (subfiles in (unique(cellinfo[,2]))){
    output_folder <- file.path(file.path(data_folder, "cicero_output"), subfiles)
    dir.create(output_folder)
  }
  file_names <- list.files(search_directory, pattern = "atac_peak_annotation\\.tsv$", full.names = TRUE)
  if (length(file_names)<1) {
    peakinfo <- c()
  } else {
    peakinfo <- read.table(file_names, header = T, sep = '\t')
  }
  #
  return(list(indata=alldata[["Peaks"]], 
              GEX=alldata[["Gene Expression"]],
              cellinfo=cellinfo, 
              peakinfo=peakinfo,
              celltype=unique(cellinfo[,2])))
}

sub_load_10X_visium_data <- function(search_directory){
  peak_counts_file <- list.files(search_directory, pattern = "\\peak_counts.rds$", full.names = TRUE)
  Peaks <- readRDS(peak_counts_file)
  peakinfo <- data.frame(peakid = rownames(Peaks))
  patient <- sub(".*(P[^/]*).*", "\\1", search_directory)
  cell_metadata <-  read_csv("/data/wcy_data/RegulationGPT_key/myocardial/scATAC_obs_dataframe.csv")
  cell_metadata <- cell_metadata[cell_metadata$patient %in% patient, ]
  if(nrow(cell_metadata) == 0)
  {
    print('Error, cellinto is NULL!')
  }
  scATAC_barcode <- sapply(cell_metadata[,1], function(x) {sub(".*#", "", x)})
  cell_metadata <- cell_metadata[scATAC_barcode %in% colnames(Peaks),]
  cell_metadata <- cell_metadata %>%
    group_by(sample) %>%
    mutate(n = n()) %>%                # 计算每个C的重复数量
    ungroup() %>%
    filter(n == max(n)) %>%            # 只保留重复数量最多的行
    select(-n)                         # 删除用于辅助的计数列
  rownames(cell_metadata)   <- sapply(cell_metadata[,1], function(x) {sub(".*#", "", x)})
  Peaks <- Peaks[,rownames(cell_metadata)]
  Peaks <- as.matrix(Peaks)
  cell_metadata <- as.data.frame(cell_metadata[colnames(Peaks),])
  cell_metadata[,1] <- colnames(Peaks)
  cell_metadata$celltype = 'Cell'
  rownames(cell_metadata) <- colnames(Peaks)
  return(list(indata=Peaks,
              GAX=NULL,
              cellinfo=cell_metadata,
              peakinfo=peakinfo,
              celltype= 'Cell',
              fitter=FALSE))
}

sub_load_pbmc_data <- function(data_folder, QC=TRUE){
  
  # 1. Read in matrix/barcode/feature data using the Matrix package
  indata <- Matrix::readMM(paste0(data_folder, "/matrix.mtx.gz")) 
  cellinfo <- read.table(paste0(data_folder, "/barcodes.tsv.gz"))
  features <- read.table(paste0(data_folder, "/features.tsv.gz"), fill = TRUE, header = FALSE)
   
  # 2. QC
  if (QC)
  {
    indata@x[indata@x > 0] <- 1 # Binarize the matrix
    non_zero_ratio <- colSums(indata != 0) / nrow(indata)
    indata <- indata[, non_zero_ratio >= 0.05]
  }
  print(dim(indata))
  
  # 3. Format cell info
  if (QC)
  {cellinfo <- data.frame(cellinfo[non_zero_ratio >= 0.05,])}
  rownames(cellinfo) <- cellinfo$V1
  names(cellinfo) <- "cells"

  # 4. Read features
  peakinfo <- features[features$V3 == "Peaks",]
  names(peakinfo) <- c("barcodes", "site_name", "types", "chr", "bp1", "bp2")
  peakinfo <- peakinfo[,c("chr", "bp1", "bp2", "site_name")]
  peakinfo$site_name <- paste(peakinfo$chr, peakinfo$bp1, peakinfo$bp2, sep="_")
  rownames(peakinfo) <- peakinfo$site_name
  
  # 5. Filter out non-peak features
  indata <- indata[features$V3 == "Peaks",]
  rownames(indata) <- rownames(peakinfo)
  colnames(indata) <- rownames(cellinfo)
  
  return(list(indata=indata, cellinfo=cellinfo, peakinfo=peakinfo))
}

### S3. calculate cicero
sub_run_cicero <- function(sub_data, output_folder, Species = 'human', window_len=5e+05){
  # Make CDS
  input_cds <-  suppressWarnings(new_cell_data_set(sub_data$indata,
                                                   cell_metadata = sub_data$cellinfo))
  input_cds <- monocle3::detect_genes(input_cds)
  #Ensure there are no peaks included with zero reads
  input_cds <- input_cds[Matrix::rowSums(exprs(input_cds)) != 0,] 
  
  
  # 3. Qauality check and Filtering
  # Visualize peak_count_per_cell
  #hist(Matrix::colSums(exprs(input_cds)))
  # Filter cells by peak_count
  # Please set an appropriate threshold values according to your data 
  max_count <-  15000
  min_count <- 2000
  input_cds <- input_cds[,Matrix::colSums(exprs(input_cds)) >= min_count] 
  #input_cds <- input_cds[,Matrix::colSums(exprs(input_cds)) <= max_count] 
  
  
  # 4. Process cicero-CDS object
  set.seed(2024)
  input_cds <- detect_genes(input_cds)
  input_cds <- estimate_size_factors(input_cds)
  input_cds <- preprocess_cds(input_cds, method = "LSI")
  input_cds <- reduce_dimension(input_cds, reduction_method = 'UMAP', 
                                preprocess_method = "LSI")
  
  #plot_cells(input_cds)   
  umap_coords <- reducedDims(input_cds)$UMAP
  cicero_cds <- make_cicero_cds(input_cds, reduced_coordinates = umap_coords)
  #saveRDS(cicero_cds, paste0(output_folder, "/cicero_cds.Rds"))
  
  if (Species == 'mouse') {
    download.file(url = "https://raw.githubusercontent.com/morris-lab/CellOracle/master/docs/demo_data/mm10_chromosome_length.txt",
                  destfile = "./mm10_chromosome_length.txt")
    chromosome_length <- read.table("./mm10_chromosome_length.txt")
  }else{
    data(human.hg19.genome)
    chromosome_length <- human.hg19.genome
  }
  
  conns <- run_cicero(cicero_cds, chromosome_length, window = window_len) # Takes a few minutes to run
  
  return(list(conns=conns,input_cds=input_cds))
}

# Main function
# You can substitute the data path below to your scATAC-seq data.
# Create a folder to save results
output_folder <- file.path(data_folder, "cicero_output")
dir.create(output_folder)

# Load data and make Cell Data Set (CDS) object 
file_type <- check_required_files_type(data_folder)
if (file_type == '10X_datasets'){
  cat(paste0("Load 10X RNA+ATAC datasets: ", data_folder, "\n"))
  sub_data <- sub_load_10X_data(data_folder)
}else if(file_type == '10X_visium'){
  cat(paste0("Load 10X Visium datasets: ", data_folder, "\n"))
  sub_data <- sub_load_10X_visium_data(data_folder)
}else if(file_type == 'pbmc'){
  sub_data <- sub_load_pbmc_data(data_folder)
}else{
  stop("The data type is not supported.")
}

# 加载注释信息
annotations <- GetGRangesFromEnsDb(ensdb = EnsDb.Hsapiens.v86)
seqlevelsStyle(annotations) <- "UCSC"
genome(annotations) <- "hg38"
if (length(select_celltype) == 0){
select_celltype <- sub_data$celltype
print(paste0('This dataset contrains the follow cells: ',sub_data$celltype))
}else{
	select_celltype <- list(select_celltype)
	print(paste0('The selected cell type is: ',select_celltype[[1]]))
}
 
for (subfiles in select_celltype){
  # 使用 tryCatch 包装整个代码块，以便捕获错误并继续
  tryCatch({
    # Run cicero
    select_cell <- sub_data$cellinfo[sub_data$cellinfo$celltype == subfiles, 1]
    if(length(select_cell) < 200){
      cat(paste0(subfiles," cell number is less than 200, skip!\n"))
      next
    }
    subcell_sub_data <- list(indata = sub_data$indata[,select_cell], 
                             cellinfo = sub_data$cellinfo[select_cell,],
                             peakinfo = sub_data$peakinfo, 
                             celltype = sub_data$celltype)
    subcell_output_folder <- file.path(output_folder, subfiles)
    result <- sub_run_cicero(subcell_sub_data, subcell_output_folder, Species = 'human')
    input_cds <- result$input_cds
    all_peaks <- row.names(exprs(input_cds))
    allpeak_file <- file.path(subcell_output_folder, "all_peaks.csv")
    write.csv(x = all_peaks, file = allpeak_file )
    write.csv(x = result$conns, file = file.path(subcell_output_folder, "cicero_connections.csv"))

    # Get ATAC GAX Matrix
    fragments.path <- list.files(data_folder, pattern = "fragments\\.tsv.gz$", full.names = TRUE)
    cat(paste0(subfiles," create Chromatin object!\n"))
    pbmc_atac <- CreateSeuratObject(counts = subcell_sub_data$indata)
    pbmc_atac[['ATAC']] <-  CreateChromatinAssay(counts = subcell_sub_data$indata, 
                                                 sep = c(":", "-"), 
                                                 genome = 'hg38',
                                                 fragments = fragments.path,
                                                 min.cells = 0, 
                                                 min.features = 0)
    cat(paste0(subfiles," Add annotations!\n"))
    DefaultAssay(pbmc_atac) <- "ATAC"
    Annotation(pbmc_atac) <- annotations
    cat("Get Gene Activity Matrix!\n")
    cat(paste0(subfiles," Get Gene Activity Matrix!\n"))
    gene_activities <- as.matrix(GeneActivity(pbmc_atac))
  
    h5_file <- file.path(subcell_output_folder, "ATAC_GAX_RNA_GEX.h5")
    print(h5_file)
    if (file.exists(h5_file)) { file.remove(h5_file)}
    h5createFile(h5_file) # 创建一个新的 HDF5 文件
    
    h5createDataset(h5_file, "RNA_GEX", dim = dim(as.matrix(sub_data$GEX[, select_cell])))
    h5write(as.matrix(sub_data$GEX[,select_cell]), h5_file, "RNA_GEX")
    h5write(colnames(sub_data$GEX[,select_cell]), h5_file, "RNA_GEX.colnames")
    h5write(rownames(sub_data$GEX[,select_cell]), h5_file, "RNA_GEX.rownames")
    
    h5createDataset(h5_file, "ATAC_GAX", dim = dim(gene_activities))
    h5write(gene_activities, h5_file, "ATAC_GAX")
    h5write(colnames(gene_activities), h5_file, "ATAC_GAX.colnames")
    h5write(rownames(gene_activities), h5_file, "ATAC_GAX.rownames")
    
    H5close()
    cat(paste0(subfiles," Save h5ad files successfully!!\n"))
    
  }, error = function(e){
    # 捕获错误并打印错误信息，继续下一个子文件的处理
    cat(paste0("Error encountered in ", subfiles, ": ", conditionMessage(e), "\n"))
  })
}



#all_peak_anns <- data.frame(pbmc_atac@assays[["ATAC"]]@annotation@elementMetadata@listData)  
#all_peak_anns <- read_csv("~/RegulationGPT/Data/PBMC/cicero_output/all_peak_anns.csv", col_names = TRUE)
#all_peak_anns <- all_peak_anns[,c('chr', 'start', 'end', 'gene_short_name')]
#colnames(all_peak_anns) <- c('chromosome', 'start', 'end', 'gene')
#input_cds <- annotate_cds_by_site(input_cds, gene_annotation_sub)
#unnorm_ga <- build_gene_activity_matrix(input_cds, conns)





