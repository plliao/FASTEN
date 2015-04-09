
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
LINKOBJS = $(SNAP)/Snap.o $(LIBDIR)/cascdynetinf.o -lrt

CC = g++
INCLUDEFLAGS = $(foreach dir,$(INCLUDEDIRS), -I $(dir))
CFLAGS = -g -Wall $(INCLUDEFLAGS)

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

.PHONY: all ctags clean show 

all: $(PROGRAM) DataMerger

$(PROGRAM): $(MAINFILE) $(OBJS) $(LINKOBJS)
	@$(CC) $(CFLAGS) $< $(OBJS) $(LINKOBJS) -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%$(SRCEXTS) $(HDRDIR)/%$(HDREXTS)
	@$(CC) $(CFLAGS) -c $< $(LINKOBJS) -o $@

DataMerger: dataMerger.cpp $(OBJS) $(LINKOBJS)
	@$(CC) $(CFLAGS) $< $(OBJS) $(LINKOBJS) -o $(addprefix $(BINDIR)/,$@)

ctags: $(MAINFILE) $(SOURCES) $(HEADERS) $(SNAPSOURCES) $(SNAPHEADERS)
	@$(CTAGS) $(CTAGFLAGS) $(MAINFILE) $(SOURCES) $(HEADERS) $(SNAPSOURCES) $(SNAPHEADERS)

clean: 
	@$(RM) $(OBJS) $(PROGRAM)

show:
	@echo 'PROGRAM     :' $(PROGRAM)
	@echo 'HEADERS     :' $(HEADERS)
	@echo 'SOURCES     :' $(SOURCES)
	@echo 'OBJS        :' $(OBJS)
	@echo 'INCLUDEFLAGS :' $(INCLUDEFLAGS)

