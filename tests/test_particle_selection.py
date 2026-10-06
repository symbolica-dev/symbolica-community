"""Verify the packaged Feynkit particle whitelist and immutable filter updates."""

import pytest
from symbolica.community import hepkit as hep


def test_particle_selection_includes_antiparticles_and_matches_complement():
    model = hep.Model.standard_model()
    process = model.process(["e-", "e+"], ["a", "a"])
    selected = process.with_filters(particle_selection=[model.particle("e-"), 22])
    diagrams = selected.generate_diagrams(progress=None)
    assert len(diagrams) == 2
    assert {d.id for d in diagrams} == {
        d.id for d in process.generate_diagrams(progress=None)
    }
    assert selected.particle_selection == [11, 22]
    assert process.particle_selection is None
    assert selected.with_filters().particle_selection == selected.particle_selection
    assert (
        len(
            selected.with_filters(particle_selection=[]).generate_diagrams(
                progress=None
            )
        )
        == 0
    )
    assert (
        len(
            selected.with_filters(particle_selection=None).generate_diagrams(
                progress=None
            )
        )
        == 2
    )


def test_particle_filters_are_mutually_exclusive_even_when_empty():
    model = hep.Model.standard_model()
    with pytest.raises(hep.GenerationError, match="mutually exclusive"):
        model.process([], [], particle_selection=[], particle_veto=[])
    selected = model.process(["e-", "e+"], ["a", "a"], particle_selection=["e-", "a"])
    with pytest.raises(hep.GenerationError, match="mutually exclusive"):
        selected.with_filters(particle_veto=[])
    switched = selected.with_filters(particle_selection=None, particle_veto=["e-"])
    assert switched.particle_selection is None
    assert len(switched.generate_diagrams(progress=None)) == 0
    with pytest.raises(hep.GenerationError):
        selected.with_filters(particle_selection=["missing_particle"])


def test_selected_loop_species_match_an_independent_particle_veto():
    model = hep.Model.standard_model()
    process = model.process(["a"], ["a"])
    options = {"loops": 1, "max_vertices": 2, "self_energy": None, "progress": None}
    selected = process.with_filters(particle_selection=["e-", "a"])
    veto = [p for p in model.particles if abs(p.pdg_code) not in {11, 22}]
    excluded = process.with_filters(particle_veto=veto)
    selected_ids = {d.id for d in selected.generate_diagrams(**options)}
    assert selected_ids
    assert selected_ids == {d.id for d in excluded.generate_diagrams(**options)}
    assert len(selected_ids) < len(process.generate_diagrams(**options))
