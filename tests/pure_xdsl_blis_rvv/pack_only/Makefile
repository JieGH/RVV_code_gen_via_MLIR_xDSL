# Issue #55: the pack-only loop. Same flags as templates/Makefile on the board.
# On a host without RVV (tests): make ARCH=   (the header then uses its plain C fallback).
# scalar_pack.inc comes from gen_scalar_inc.py (host side); the series script makes it before copying.
CXX = g++
CC = gcc
ARCH ?= -march=rv64gcv -mabi=lp64d
CXXFLAGS = -O2 -std=c++17 $(ARCH) -fno-strict-aliasing -I../templates
CFLAGS = -O3 $(ARCH) -fno-strict-aliasing

pack_b_only: pack_b_only.cpp model_level.o scalar_pack.inc ../templates/pack_b_variants.h
	$(CXX) $(CXXFLAGS) -o $@ pack_b_only.cpp model_level.o -lm

model_level.o: ../modelLevel/model_level.c ../modelLevel/model_level.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f pack_b_only model_level.o
