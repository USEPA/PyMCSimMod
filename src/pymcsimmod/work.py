from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify
from rich import print

import pymcsimmod.model as model
from pymcsimmod.parser import ModelParser


def bottles(num: int, beverage: str):
    for i in range(num, 0, -1):
        typer.secho(
            f"{i} bottles of {beverage} on the wall, {i} bottles of {beverage}, take one down...",
            fg="green",
        )
        time.sleep(random())
    typer.secho(f"No more bottles of {beverage} on the wall!", fg="green")


def super_add(a: int, b: int) -> int:
    # a silly function for an example unit-test
    return a + b
