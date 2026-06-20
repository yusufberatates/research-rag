"""Pre-seeded top-level fields for the quantum-radar / quantum-sensing corpus.

The taxonomy is four levels deep: Main Field -> Subfield -> Sub-subfield ->
Papers. Only the seven main fields below are pre-seeded (with multi-paragraph
descriptors written by hand); subfields and sub-subfields are grown
automatically by the classifier as papers are ingested.

Routing and classification read these descriptors, so they are intentionally
substantial (2-3 paragraphs) and describe both what belongs in the field and
what does NOT, to keep the LLM's top-down choices crisp.
"""
from __future__ import annotations

SEED_FIELDS: dict[str, str] = {
    "quantum_illumination_theory": (
        "Quantum illumination is the theory of detecting a low-reflectivity "
        "target embedded in bright thermal noise by exploiting entanglement "
        "between a transmitted 'signal' mode and a retained 'idler' mode. "
        "Work in this field covers Lloyd's original discrete-variable "
        "proposal, the Gaussian-state (two-mode squeezed vacuum) formulation "
        "of Tan et al., and the 6 dB error-exponent advantage over the best "
        "classical coherent-state transmitter of equal energy.\n\n"
        "Core topics include the quantum Chernoff bound and its application "
        "to target-detection error exponents, the role of signal-idler "
        "correlations that survive even when entanglement itself does not, "
        "and the receivers (optical parametric amplifier, phase-conjugate) "
        "that approach the theoretical advantage. The field is concerned with "
        "fundamental performance limits and information-theoretic bounds, not "
        "with specific hardware implementations.\n\n"
        "This field is the theoretical foundation that microwave_quantum_radar "
        "and quantum_lidar build physical systems around. Papers belong here "
        "when their primary contribution is the theory, bounds, or receiver "
        "design for illumination-style detection rather than a device, a "
        "metrological estimation protocol, or a propagation study."
    ),
    "microwave_quantum_radar": (
        "Microwave quantum radar concerns the physical realization of "
        "quantum-enhanced detection and ranging in the microwave band, where "
        "practical radar operates. Because microwave photons cannot be "
        "entangled directly at useful powers, this field centers on "
        "Josephson-junction devices: Josephson parametric amplifiers (JPA) "
        "and Josephson travelling-wave parametric amplifiers (JTWPA) that "
        "generate two-mode squeezed vacuum at microwave frequencies, and the "
        "cryogenic systems required to operate them.\n\n"
        "Key subjects include microwave entanglement generation and "
        "verification, the microwave-to-optical transduction needed to store "
        "or process an idler, demonstrations of microwave quantum "
        "illumination, and the severe practical limits (cryogenic idler "
        "storage, low photon numbers, short range) that separate laboratory "
        "demonstrations from deployable radar.\n\n"
        "Papers belong here when the contribution is a microwave-band device, "
        "experiment, or system. Pure detection theory belongs in "
        "quantum_illumination_theory; optical-band ranging belongs in "
        "quantum_lidar; amplifier physics with no radar framing may belong in "
        "supporting_quantum_optics."
    ),
    "rydberg_atomic_receivers": (
        "Rydberg atomic receivers use atoms excited to high principal quantum "
        "number (Rydberg states) as sensitive, self-calibrated detectors of "
        "radio-frequency and microwave electric fields. Detection relies on "
        "electromagnetically induced transparency (EIT) and the Autler-Townes "
        "splitting that an applied RF field imposes on the Rydberg "
        "transition, read out optically through a vapor cell.\n\n"
        "Topics include SI-traceable field sensing, broadband tunability "
        "across the RF spectrum, the standard-quantum-limit and atomic "
        "projection-noise limits on sensitivity, antenna-free reception, and "
        "the use of Rydberg sensors as receivers for communication and "
        "radar-return signals. The field bridges atomic physics and "
        "RF engineering.\n\n"
        "Papers belong here when the sensing element is an atomic/Rydberg "
        "medium. Superconducting or parametric microwave detection belongs in "
        "microwave_quantum_radar; underlying atom-light physics with no "
        "receiver framing belongs in supporting_quantum_optics."
    ),
    "quantum_lidar": (
        "Quantum LiDAR covers quantum-enhanced ranging, imaging, and "
        "detection at optical and near-infrared wavelengths. It includes "
        "entangled-photon and single-photon LiDAR, quantum pulse-compression "
        "and quantum-correlated ranging, ghost imaging, and photon-counting "
        "schemes that improve range resolution, timing precision, or "
        "noise rejection over classical optical LiDAR.\n\n"
        "Central concerns are timing/range precision beyond the classical "
        "limit, robustness in photon-starved or high-background conditions, "
        "and the entanglement or squeezing resources that provide the "
        "advantage. The optical band distinguishes it physically from "
        "microwave_quantum_radar even when the underlying illumination "
        "theory is shared.\n\n"
        "Papers belong here for optical/IR ranging and imaging systems. "
        "Shared detection bounds belong in quantum_illumination_theory; "
        "general squeezed-light or SPDC source physics without a ranging "
        "application belongs in supporting_quantum_optics."
    ),
    "quantum_detection_theory": (
        "Quantum detection and estimation theory provides the mathematical "
        "backbone for all quantum sensing and radar: quantum hypothesis "
        "testing (Helstrom bound, quantum Chernoff and Bhattacharyya bounds), "
        "quantum and classical Fisher information, the quantum Cramer-Rao "
        "bound, and multi-parameter estimation. It formalizes the limits on "
        "distinguishing quantum states and on estimating parameters encoded "
        "in them.\n\n"
        "Topics include single- and multi-parameter metrology, distributed "
        "and networked sensing with entangled resources, privacy and "
        "robustness of sensing protocols, optimal measurements and "
        "estimators, and the scaling of precision (standard quantum limit vs. "
        "Heisenberg limit) with resources. This field supplies the bounds "
        "that target-detection and ranging papers invoke.\n\n"
        "Papers belong here when the contribution is a general detection or "
        "estimation result, bound, or protocol. Illumination-specific "
        "detection theory may instead fit quantum_illumination_theory; a "
        "physical device belongs in one of the hardware fields."
    ),
    "supporting_quantum_optics": (
        "Supporting quantum optics collects the enabling physics that quantum "
        "radar and sensing depend on but which is not itself a detection or "
        "ranging result: spontaneous parametric down-conversion (SPDC) and "
        "other entangled-photon sources, squeezed light generation, "
        "two-mode squeezed vacuum, optical and microwave parametric "
        "amplification, quantum-optical transducers, and circuit-QED "
        "entanglement generation.\n\n"
        "It also covers loss and decoherence models, mode and photon-number "
        "statistics, and the characterization and verification of "
        "non-classical states. These are the building blocks reused across "
        "illumination theory, microwave radar, LiDAR, and atomic receivers.\n\n"
        "Papers belong here when the focus is the source, amplifier, "
        "transducer, or state physics in its own right. If the same physics "
        "is presented as a radar/LiDAR/receiver system or as a detection "
        "bound, prefer the corresponding application field."
    ),
    "critical_assessment_literature": (
        "Critical assessment literature evaluates whether quantum radar and "
        "quantum sensing deliver a real advantage in practice. It includes "
        "review and perspective articles, feasibility and range-limitation "
        "analyses, comparisons against optimized classical radar, and "
        "skeptical or debunking treatments of overstated claims.\n\n"
        "Topics include the loss of entanglement advantage under realistic "
        "noise and loss, integration-time and cryogenic-overhead penalties, "
        "the gap between 6 dB error-exponent advantages and usable "
        "detection range, and roadmaps or meta-analyses of the field's "
        "maturity.\n\n"
        "Papers belong here when their main contribution is critique, "
        "review, or feasibility assessment rather than a new device, theory, "
        "or protocol. A paper that proposes a method and only briefly "
        "discusses limitations belongs in its method's field instead."
    ),
}


# Tier of each top-level field: 1 = core quantum radar (illumination, microwave
# radar, Rydberg receivers, LiDAR, detection theory), 2 = supporting physics,
# 3 = adjacent / commentary. New (non-seeded) fields default to tier 3.
TIER_BY_FIELD: dict[str, int] = {
    "quantum_illumination_theory": 1,
    "microwave_quantum_radar": 1,
    "rydberg_atomic_receivers": 1,
    "quantum_lidar": 1,
    "quantum_detection_theory": 1,
    "supporting_quantum_optics": 2,
    "critical_assessment_literature": 3,
}
DEFAULT_TIER = 3


def tier_for_field(field: str) -> int:
    """Tier (1/2/3) implied by a paper's assigned main field."""
    return TIER_BY_FIELD.get((field or "").strip().lower(), DEFAULT_TIER)


def seeded_tree() -> dict:
    """Return a fresh taxonomy dict pre-populated with the seven main fields
    (each with an empty set of subfields)."""
    return {
        "fields": {
            name: {"descriptor": descriptor, "subfields": {}}
            for name, descriptor in SEED_FIELDS.items()
        }
    }
