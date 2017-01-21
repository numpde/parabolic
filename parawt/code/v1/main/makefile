
default:
	@echo Nothing to do here. Try to: cat makefile

backupmakefile:
	@cp -v makefile ./makefiles/makefile_previous

hpc2: backupmakefile
	@cp -v ./makefiles/makefile_hpc2 makefile

rioc: backupmakefile
	@cp -v ./makefiles/makefile_rioc makefile

ramac: backupmakefile
	@cp -v ./makefiles/makefile_ramac makefile
