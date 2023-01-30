sqlfile_path <- "GEOmetadb.sqlite"
out_file_path = commandArgs(trailingOnly = TRUE)[1]

if (!file.exists(out_file_path)) {
  # Loading here because it takes awhile to load
  #   so only want to load it if file not yet created.
  library("GEOmetadb")

  sqlfile <- getSQLiteFile(destdir = "/Data", destfile = paste0(sqlfile_path, ".gz"))

  con <- dbConnect(SQLite(), paste0("/Data/", sqlfile_path))

  gse <- dbGetQuery(con,'select * from gse')[,c("gse", "title", "summary")]

  dbDisconnect(con)

  write.table(gse, out_file_path, sep="\t", quote=FALSE, row.names=FALSE, col.names=TRUE, fileEncoding="UTF-8")

  unlink(sqlfile)
}
