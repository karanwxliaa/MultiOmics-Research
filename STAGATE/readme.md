For the SlideSeq data of the Mouse Olfactory bulb, the following changes were made
Data: Change the file *Puck_200127_15_bead_locations.xlsx* by
      1. Drop col 1 contaning S.No.
      2. Move the *Barcode* column from the 3rd to 1st column (To use barcodes as labels)
Code:
      Several architectural changes to incorporate TFv2
      Change the .toarrey() in line 58 of train_STAGATE.py file
      Add *tf.compat.v1.disable_eager_execution()* to avoid eager execution erros
      A few other changes with packages and dependencies can be solved on the spot
