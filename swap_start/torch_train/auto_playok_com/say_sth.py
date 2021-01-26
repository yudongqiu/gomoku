#!/usr/bin/env python
# coding: utf-8
import sys
import pyautogui

s = ''.join(sys.argv[1:])

pyautogui.PAUSE = 0.1
pyautogui.click((3300,560))
pyautogui.click((3300,835))
pyautogui.typewrite(s)
pyautogui.press("enter")
