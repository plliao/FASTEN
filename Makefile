
SnapDirPath = ../Snap-2.3
GLIB = $(SnapDirPath)/glib-core
SNAP = $(SnapDirPath)/snap-core
SNAPADV = $(SnapDirPath)/snap-adv
SNAPEXP = $(SnapDirPath)/snap-exp

BINDIR = bin
SRCDIR = src
LIBDIR = lib
HDRDIR = include
OBJDIR = obj
SNAPLIBDIRS = $(GLIB) $(SNAP) $(SNAPADV) $(SNAPEXP)
INCLUDEDIRS = $(HDRDIR) $(SNAPLIBDIRS)
#LINKOBJS = $(SNAP)/Snap.o $(SNAPADV)/cascdynetinf.o $(SNAPADV)/kronecker.o
LINKOBJS = $(LIBDIR)/Snap.o $(LIBDIR)/cascdynetinf.o $(LIBDIR)/kronecker.o

CC = g++
INCLUDEFLAGS = $(foreach dir,$(INCLUDEDIRS), -I $(dir))
#CFLAGS = -g -Wall -ffast-math -fopenmp $(INCLUDEFLAGS)
CFLAGS = -O3 -Wall -ffast-math -fopenmp $(INCLUDEFLAGS)

CTAGS = ctags
CTAGFLAGS = 

RM = rm -f

SRCEXTS = .cpp
HDREXTS = .h

SOURCES = $(wildcard $(addprefix $(SRCDIR)/*, $(SRCEXTS)))
HEADERS = $(wildcard $(addprefix $(HDRDIR)/*, $(HDREXTS)))
OBJS = $(addprefix $(OBJDIR)/, $(addsuffix .o, $(notdir ,$(basename $(SOURCES)))))
SNAPSOURCES = $(foreach dir,$(SNAPLIBDIRS),$(wildcard $(addprefix $(dir)/*, $(SRCEXTS))))
SNAPHEADERS = $(foreach dir,$(SNAPLIBDIRS),$(wildcard $(addprefix $(dir)/*, $(HDREXTS))))

FILES = $(wildcard $(addprefix *, $(SRCEXTS)))
PROGRAMS = $(addprefix $(BINDIR)/, $(basename $(FILES)))
PROGRAMSERROR = $(addprefix $(BINDIR)/, $(addsuffix .Err, $(basename $(FILES))))

.PHONY: all clean show 

all: $(PROGRAMS)

.PRECIOUS: $(OBJDIR)/%.o
$(OBJDIR)/%.o: $(SRCDIR)/%$(SRCEXTS) $(HDRDIR)/%$(HDREXTS)
	@$(CC) $(CFLAGS) -c $< -lrt -o $@

$(BINDIR)/%: %$(SRCEXTS) $(OBJS) $(LINKOBJS)
	@$(CC) $(CFLAGS) $< $(OBJS) $(LINKOBJS) -lrt -o $@

ctags: $(SOURCES) $(HEADERS) $(SNAPSOURCES) $(SNAPHEADERS)
	@$(CTAGS) $(CTAGFLAGS) $(SOURCES) $(HEADERS) $(SNAPSOURCES) $(SNAPHEADERS)

clean: 
	@$(RM) $(OBJS) $(PROGRAMS) $(PROGRAMSERROR)

show:
	@echo 'FILES           :' $(FILES)
	@echo 'PROGRAMS        :' $(PROGRAMS)
	@echo 'HEADERS         :' $(HEADERS)
	@echo 'SOURCES         :' $(SOURCES)
	@echo 'OBJS            :' $(OBJS)
	@echo 'INCLUDEFLAGS    :' $(INCLUDEFLAGS)

