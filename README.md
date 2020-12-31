# Circle Slice Flows and the Variational Determinant Estimator

This repository contains the codebase related to my master thesis [Circle Slice Flows and the Variational Determinant Estimator](https://github.com/P4ppenheimer/master_thesis) under supervision of [Emiel Hoogeboom](https://scholar.google.nl/citations?user=nkTd_BIAAAAJ&hl=en) and [Stratis Gavves](https://scholar.google.com/citations?user=QqfCvsgAAAAJ&hl=en) prepared during 2020 at the University of Amsterdam (UvA). 

The thesis introduces _Circle Slice Flows_ and [_The Variational Determinant Estimator_](https://arxiv.org/abs/2012.13311). The former is a novel type of flow that allows density estimation on the D dimensional hypersphere, and the latter is a variational extension of the recently introduced determinant estimator by [Sohl-Dickstein (2020)](https://arxiv.org/abs/2005.06553v2), which utilizes spherical flows to model a proposal distribution for variance reduction. 

The repository contains an implementation of both concepts and the recently introduced family of _Cylindrical Flows_ introduced by 
[Rezende et al. (2020)](https://arxiv.org/abs/2002.02428). Additionally, the repository contains an implementation of Neural Spline Flows, [Durkan et al. (2019)](https://arxiv.org/abs/1906.04032) , based [on](https://github.com/bayesiains/nflows/tree/master/nflows/transforms/splines), and the Power Spherical Distribution, [De Cao, Aziz (2020)](https://arxiv.org/abs/2006.04437), based [on](https://github.com/nicola-decao/power_spherical). Furthermore, it contains a first building blocks of the [hyperspherical VAE](https://arxiv.org/abs/1804.00891) based [on](https://github.com/nicola-decao/s-vae-pytorch) such that Circle Slice and Cylindrical flows can be used to model the variational posterior in such a setting. But this has not been extensively used.

It is still subject to change and the readme will be extended and clarified at a later time point. If you have any questions, feel free to reach out to me via simon.passenheim@gmail.com.

