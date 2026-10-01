# Independent GPU evidence replay

These validators executed read-only after the two exact launch commits. They do not load model weights, invoke Modal, or read evaluator labels. Source snapshots bind the strict parser to the actual initial and candidate launch commits, not the later branch head.

The saved result package contains initial_run/, v4_run/, candidate_inputs/, and validation/tokenizer/. Install tokenizers==0.21.1 and Jinja2==3.1.6 in a separate environment. Prepare a scratch replay layout with sibling review_initial/, review_v4/, qanta-buzzer/, initial_run/, and v4_run/ directories. Copy the two validator folders here into their corresponding review folders, including remote_sources.json. Put the packaged tokenizer assets at review_initial/tokenizer/.

Use a source checkout or symlink named qanta-buzzer/. Restore the immutable v3 input-freeze archive's modal_pilot_data/ there, and copy candidate_inputs/ as modal_pilot_v4_data/. Place the two retained run directories at their sibling locations. Initial validator uses these fixed relative locations; v4 also accepts explicit --run, --data, --assets, --prior, and --out paths.

Run python review_initial/validate_initial.py and python review_v4/validate_v4.py --run v4_run. These write fresh machine reports beside their validators unless v4 --out is specified. Original checks passed 72 aggregate plus 4,956 row checks for v3, and 77 aggregate plus 2,655 row checks for v4. The original model weights were checked against official pinned Hub LFS metadata, not independently downloaded locally.

Recorded initial source: d4105ca1f3a705611c1e65076723838ee67cc0df. Recorded v4 source: a2013af95f7fa49f055423a9b398358aaf252c64. There were 354+177=531 GPU development responses, and no main or choices-only inference. Costs remain configured-resource estimates, not verified invoices. This evidence addition changes no runtime source or launch workflow and does not enable another allocation.
