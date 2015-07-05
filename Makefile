
SnapDirPath = ../Snap-2.3
GLIB = $(SnapDirPath)/glib-core
SNAP = $(SnapDirPath)/snap-core
SNAPADV = $(SnapDirPath)/snap-adv
SNAPEXP = $(SnapDirPath)/snap-exp


PROGRAMNAME = RNCascade

BINDIR = bin
SRCDIR = src
LIBDIR = lib
HDRDIR = include
OBJDIR = obj
SNAPLIBDIRS = $(GLIB) $(SNAP) $(SNAPADV) $(SNAPEXP)
INCLUDEDIRS = $(HDRDIR) $(SNAPLIBDIRS)
LINKOBJS = $(LIBDIR)/Snap.o $(LIBDIR)/cascdynetinf.o $(LIBDIR)/kronecker.o

CC = g++
INCLUDEFLAGS = $(foreach dir,$(INCLUDEDIRS), -I $(dir))
#CFLAGS = -g -Wall -ffast-math -fopenmp $(INCLUDEFLAGS)
CFLAGS = -O3 -Wall -ffast-math -fopenmp $(INCLUDEFLAGS)

CTAGS = ctags
CTAGFLAGS = 

RM = rm

SRCEXTS = .cpp
HDREXTS = .h

PROGRAM = $(addprefix $(BINDIR)/, $(PROGRAMNAME))
SOURCES = $(wildcard $(addprefix $(SRCDIR)/*, $(SRCEXTS)))
HEADERS = $(wildcard $(addprefix $(HDRDIR)/*, $(HDREXTS)))
OBJS = $(addprefix $(OBJDIR)/, $(addsuffix .o, $(notdir ,$(basename $(SOURCES)))))
SNAPSOURCES = $(foreach dir,$(SNAPLIBDIRS),$(wildcard $(addprefix $(dir)/*, $(SRCEXTS))))
SNAPHEADERS = $(foreach dir,$(SNAPLIBDIRS),$(wildcard $(addprefix $(dir)/*, $(HDREXTS))))
MAINFILE = main.cpp

UTILITYFILES = $(filter-out $(MAINFILE), $(wildcard $(addprefix *, $(SRCEXTS)))) 
UTILITYPROGRAMS = $(addprefix $(BINDIR)/, $(basename $(UTILITYFILES)))

.PHONY: all ctags clean show 

all: $(PROGRAM) $(UTILITYPROGRAMS) ctags

$(PROGRAM): $(MAINFILE) $(OBJS) $(LINKOBJS)
	@$(CC) $(CFLAGS) $< $(OBJS) $(LINKOBJS) -lrt -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%$(SRCEXTS) $(HDRDIR)/%$(HDREXTS)
	@$(CC) $(CFLAGS) -c $< $(LINKOBJS) -lrt -o $@

$(BINDIR)/%: %$(SRCEXTS) $(OBJS) $(LINKOBJS)
	@$(CC) $(CFLAGS) $< $(OBJS) $(LINKOBJS) -lrt -o $@

ctags: $(MAINFILES) $(MAINFILE) $(SOURCES) $(HEADERS) $(SNAPSOURCES) $(SNAPHEADERS)
	@$(CTAGS) $(CTAGFLAGS) $(MAINFILES) $(MAINFILE) $(SOURCES) $(HEADERS) $(SNAPSOURCES) $(SNAPHEADERS)

clean: 
	@$(RM) $(OBJS) $(PROGRAM)

show:
	@echo 'PROGRAM         :' $(PROGRAM)
	@echo 'MAINFILE        :' $(MAINFILE)
	@echo 'UTILITY FILES   :' $(UTILITYFILES)
	@echo 'UTILITY PROGRAMS:' $(UTILITYPROGRAMS)
	@echo 'HEADERS         :' $(HEADERS)
	@echo 'SOURCES         :' $(SOURCES)
	@echo 'OBJS            :' $(OBJS)
	@echo 'INCLUDEFLAGS    :' $(INCLUDEFLAGS)

