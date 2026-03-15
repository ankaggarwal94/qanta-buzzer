# Poster readability improvement
_Exported on 3/11/2026 at 14:27:50 PDT from Cursor (2.6.18)_

---

**User**

Given the following files as reference, improve generate_poster.py so that the poster contains all the same content but is actually readable:

Directly relevant files are:

- [generate_poster.py](/Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer/generate_poster.py)
- [poster.pdf](/Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer/generated/quizbowl_mc_stopping_data_driven/poster.pdf)
- [poster.png](/Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer/generated/quizbowl_mc_stopping_data_driven/poster.png)
- [evaluation_report.json](/Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer/artifacts/smoke/evaluation_report.json)

Also directly used as poster content/visual inputs by the current generator:

- [generate_presentation.py](/Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer/generate_presentation.py)
- [frames](/Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer/generated/quizbowl_mc_stopping_data_driven/frames)

Relevant specification/guidance files for poster requirements:

- [ed_threads.md](/Users/ankit.aggarwal/Dropbox/Stanford/CS234/ed_threads.md)
- [ed_threads.json](/Users/ankit.aggarwal/Dropbox/Stanford/CS234/ed_threads.json)
- [google_doc_poster_guidelines.txt](/Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer/poster_chatgpt_bundle_20260311/poster_guidelines/google_doc_poster_guidelines.txt)

If you want the strictly minimal set for “what files affect the rendered poster right now,” it’s just:
- [generate_poster.py](/Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer/generate_poster.py)
- [evaluation_report.json](/Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer/artifacts/smoke/evaluation_report.json)
- [poster.pdf](/Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer/generated/quizbowl_mc_stopping_data_driven/poster.pdf)
- [poster.png](/Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer/generated/quizbowl_mc_stopping_data_driven/poster.png)

---

**Cursor**

Let me start by reading the key files to understand the current poster generator and requirements.

Let me check what fonts are available on this macOS system, and also look at the existing poster more carefully.

Now I see the root cause clearly. The current script only has Linux font paths (`/usr/share/fonts/truetype/dejavu/`). On this macOS system those don't exist, so **every single font falls back to PIL's default 10px bitmap font**. That's why the entire poster is unreadable.

macOS has Arial at `/System/Library/Fonts/Supplemental/Arial.ttf` and `Arial Bold.ttf`, which I'll use.

Now let me write the improved poster generator. The key fixes:

1. **Cross-platform fonts** -- add macOS Arial paths alongside Linux DejaVu
2. **3-column layout** -- cleaner than the current complex 3-row grid
3. **Larger minimum font sizes** -- never below ~24px (current code goes down to 10-14px)
4. **Better spacing** -- more room to breathe in each panel
5. **All original content preserved**

Now let me run the poster generator and check the output.

Massive improvement - the poster is now clearly readable with proper Arial fonts, all content preserved, and a clean 3-column layout. Let me check for linter errors.

All minor warnings -- let me clean up the unused variables.

Lines 472 and 535 have unused `y1`. Let me fix those.

All done. Here's a summary of the changes:

**Root cause**: The original script only had Linux font paths (`/usr/share/fonts/truetype/dejavu/`). On macOS, none of those exist, so every font fell back to PIL's 10px default bitmap font -- making the entire poster unreadable.

**Key fixes in the rewrite**:

1. **Cross-platform fonts** -- Added macOS paths (`/System/Library/Fonts/Supplemental/Arial.ttf`, `Arial Bold.ttf`) plus Helvetica TTC fallback, alongside the existing Linux paths. Now finds Arial on your system immediately.

2. **3-column layout** instead of the original complex 3-row grid:
   - Column 1: Problem + Method + Conclusions + References
   - Column 2: Running Example (full height, with adaptive spacing)
   - Column 3: Smoke Snapshot + Controls + Limitations

3. **Larger minimum font sizes** -- The original used `min_size` values as low as 10-14px throughout `draw_text_fit` calls. Now minimums are 18-24px, ensuring readability at poster-session distance (~4 feet).

4. **Taller card headers** -- Increased from 62px to 76px so section titles have breathing room.

5. **Adaptive spacing** -- The example card distributes extra vertical space as inter-section padding, so the poster feels spacious rather than cramped.

6. **All content preserved** -- Every bullet point, metric chip, chart, clue card, control result, and reference from the original is present.

