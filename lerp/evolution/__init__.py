"""A closed evolutionary learning loop: breed adapters, let the children learn verifiable problems, select on an independent test.

The pieces: ``problems`` (generators with exact verifiers), ``learning`` (LoRA training that continues from an inherited adapter, and rank
compression so inheritance does not grow the adapter), ``archive`` (organisms, lineage, niches), ``orchestrator`` (the generation loop and the
controls it is compared against).
"""
