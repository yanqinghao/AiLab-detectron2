#!/bin/bash

pyarmor obfuscate --exclude detectron2,fvcore,cocoapi --recursive run.py
