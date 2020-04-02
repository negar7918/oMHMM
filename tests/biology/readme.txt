To run the python test, you need to upload the needed data file and run the below steps. The file can be requested from the authors of the following article:

Leung, M., Davis, A., Gao, R., Casasent, A., Wang, Y., Sei, E., ..., Navin, N.:
Single-cell  dna  sequencing  reveals  a  late-dissemination  model  in  metastatic  colorectalcancer.
Genome Research 27(8), 1287–1299 (2017)

When requesting it, ask for the count and ratio data regarding CRC2 and then call them file_count and file_ratio in R after reading them.

Then perform the following in R:

1. Remove some cells according to the article:
    file_count_removed_cells <- subset(file_count, select = -c(MA.58_, MA.60_, MA.68_, MA.69_, MA.70_, PA.12_, PA.15_, PA.4_, PA.9_))
    file_ratio_removed_cells <- subset(file_ratio, select = -c(MA.58_, MA.60_, MA.68_, MA.69_, MA.70_, PA.12_, PA.15_, PA.4_, PA.9_))

2. Perform the following preprocessing in R:
    med <- apply(file_count_removed_cells[,4:39],2, FUN = "median")
    corrected_counts <- file_ratio_removed_cells
    corrected_counts[,4:39] <- ratio_2_removed_cells[,4:39] * med

3. Fetch chromosome 4 and save it to the file in R:
    chrom_4 <- subset(corrected_counts, chrom == 4)
    write.csv(round(chrom_4[,4:39]) , file = paste0("./chrom4.csv"))

4. Note that the produced file "chrom4.csv" should be located in the biology folder to be able to run the test.


##########################################################################################################

How to run the python test:

To see the plots, uncomment the commented lines (not all at the same time) or simply look at the attached eps files in the biology folder.
