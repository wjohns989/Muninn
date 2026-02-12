# Muninn Branding Assets

This directory contains the visual identity assets for the Muninn project.

## Logo Concept

**Muninn** (Old Norse: "Memory") - One of Odin's ravens who flies across the world gathering knowledge.

## Design Philosophy

- **Minimalist Geometric:** Clean, precise forms
- **Old Viking:** Ancient runestones and sacred geometry
- **Runic Integration:** Authentic Elder Futhark scripts
- **Dark Mode First:** Optimized for modern developer environments

## Color Palette

```
Midnight Blue:  #0B1426  (primary background)
Raven Black:    #1A1A1D  (deep elements)
Runic Silver:   #C0C5CE  (highlights)
Ethereal Blue:  #4A6FA5  (accents)
```

## Files

- `muninn_tray_icon.png` - System tray icon (Norse raven with purple crystal wings)
- `muninn_banner.png` - Project banner (full-resolution marketing asset)

> **Note**: Image files are gitignored due to size. Place your copies in this directory after cloning.
> The tray application will auto-detect `muninn_tray_icon.png` from this `assets/` directory.

## Usage

The tray application (`tray_app.py`) searches for icons in this order:
1. `assets/muninn_tray_icon.png`
2. `assets/muninn.ico`
3. `assets/muninn.png`
4. Falls back to a generated Mannaz rune icon with status-colored indicators
