---
author: Wes Barnett, Vedran Miletić
---

# GROMACS tutorials

!!! note
    The GROMACS tutorials that follow were written in Markdown by
    [Wes Barnett](https://github.com/wesbarnett) and were originally published
    in the tutorials section of his former website. The tutorials are presently
    maintained by [Vedran Miletić](https://vedran.miletic.net/) and published in
    the tutorials section of the [Group for Applications and Services on Exascale
    Research Infrastructure (GASERI) website](../../index.md).

These are some [GROMACS](https://www.gromacs.org/) tutorials for
beginners that I wrote during my Ph.D. work. I found that many GROMACS
tutorials focused on biomolecules such as proteins, where few focused
on simple systems. I hope these tutorials fill the gap and give you a
greater understanding of how to set up and run simple molecular
simulations and how to utilize advanced sampling techniques.

It's not necessary to do
the tutorials in order, but the first two tutorials are essential
before going on to the others, as the structure file (`methane.pdb`)
and the topology file (`topol.top`) for methane from tutorial 2 are used
in all subsequent tutorials. The tutorials are designed for GROMACS
version 5.1 and up. If you are using an older version, some of the
commands or parameters may have changed. Note especially that the pull
code for umbrella sampling has changed since 5.0 and older releases.

## Prerequisites

I assume you have some working knowledge of the command line (*e.g.*,
bash). Specifically, you should know how to make directories, change your current
directory into them, edit text files, and download files to your system. When
you see a `$` or `>` this is the prompt on the command line and
indicates you should type the text following it. If the command line
is new to you, consider searching for a tutorial.

I also assume you have GROMACS installed on a machine available to you. Source
code and installation instructions can be found on the [GROMACS documentation
page](https://manual.gromacs.org/documentation/).

Throughout the tutorials, we'll be using OPLS methane and TIP4PEW water.

## Contents

1. [Water](1-tip4pew-water/index.md) -- Basics of setting up a simulation. Find
   out the density of TIP4PEW water.
2. [One methane in water](2-methane-in-water/index.md) -- How to create a
   topology file for a molecule and solvate it. Get the radial distribution
   function.
3. [Several methanes in water](3-methanes-in-water/index.md) -- How to put
   multiple solutes into a system. Get the methane-methane potential of mean
   force.
4. [Free energy of solvation of methane](4-methane-fe/index.md) -- How to do a
   free energy simulation when coupling a molecule. Use MBAR to get the result.
5. [Window sampling](5-umbrella/index.md) -- How to get methane-methane PMF from
   window sampling using pull code.
6. [Test particle insertion](6-tpi/index.md) -- How to get the excess chemical
   potential of methane using test particle insertion.

## Questions?

See [my contact page](https://vedran.miletic.net/#contact).
