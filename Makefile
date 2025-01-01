.PHONY: all clean
CC=gcc
CFLAGS=-c -O3 -Wall -Wno-switch
LDFLAGS=
SRCDIR=src
BUILDDIR=build
TARGET=$(BUILDDIR)/cudarrows.exe
SOURCES=$(wildcard $(SRCDIR)/*.c) $(wildcard $(SRCDIR)/*.cu)
OBJECTS=$(subst $(SRCDIR)/,$(BUILDDIR)/,$(patsubst %.cu,%.o,$(patsubst %.c,%.o,$(SOURCES))))

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf $(BUILDDIR)/*