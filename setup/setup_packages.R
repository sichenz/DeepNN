get_libraries <- function(filenames_list) { 
  lapply(filenames_list,function(thelibrary){    
    if (do.call(require,list(thelibrary)) == FALSE) 
      do.call(install.packages,list(thelibrary,repos = "http://cran.us.r-project.org")) 
    do.call(library,list(thelibrary))
  })
}

libraries_used=c("bit64","magrittr","data.table","foreach",'doParallel','bayesm','glue','reshape2','pracma','argparse','yaml','ggplot2','ggpubr','gridExtra','testthat','reticulate','car')

get_libraries(libraries_used)