# From Geant-4.9.4_p4, G4OpBoundaryProcess
# the values are assigned to variable theStatus
G4OpBP = ("Undefined", "Transmission", "FresnelRefraction",
          "FresnelReflection", "TotalInternalReflection",
          "LambertianReflection", "LobeReflection",
          "SpikeReflection", "BackScattering",
          "Absorption", "Detection", "NotAtBoundary",
          "SameMaterial", "StepTooSmall", "NoRINDEX",
          "PolishedLumirrorAirReflection",
          "PolishedLumirrorGlueReflection",
          "PolishedAirReflection",
          "PolishedTeflonAirReflection",
          "PolishedTiOAirReflection",
          "PolishedTyvekAirReflection",
          "PolishedVM2000AirReflection",
          "PolishedVM2000GlueReflection",
          "EtchedLumirrorAirReflection",
          "EtchedLumirrorGlueReflection",
          "EtchedAirReflection",
          "EtchedTeflonAirReflection",
          "EtchedTiOAirReflection",
          "EtchedTyvekAirReflection",
          "EtchedVM2000AirReflection",
          "EtchedVM2000GlueReflection",
          "GroundLumirrorAirReflection",
          "GroundLumirrorGlueReflection",
          "GroundAirReflection",
          "GroundTeflonAirReflection",
          "GroundTiOAirReflection",
          "GroundTyvekAirReflection",
          "GroundVM2000AirReflection",
          "GroundVM2000GlueReflection", "Dichroic")
G4OpBP_enum = {value: i for i, value in enumerate(G4OpBP)}
G4OpProcessSubType = {
    31: "fOpAbsorption",
    32: "fOpBoundary",
    33: "fOpRayleigh",
    34: "fOpWLS",
    35: "fOpMieHG",
    91: "TRANSPORTATION",
    92: "COUPLED_TRANSPORTATION",
    401: "STEP_LIMITER",
    402: "USER_SPECIAL_CUTS",
    403: "NEUTRON_KILLER"
}

# G4EmProcessSubType.hh File Reference
G4EmProcessSubType = {
   "CoulombScattering": 1,
   "Ionisation": 2,
   "Bremsstrahlung": 3,
   "PairProdByCharged": 4,
   "Annihilation": 5,
   "AnnihilationToMuMu": 6,
   "AnnihilationToHadrons": 7,
   "NuclearStopping": 8,

   "MultipleScattering": 10,

   "Rayleigh": 11,
   "PhotoElectricEffect": 12,
   "ComptonScattering": 13,
   "GammaConversion": 14,
   "GammaConversionToMuMu": 15,

   "Cerenkov": 21,
   "Scintillation": 22,
   "SynchrotronRadiation": 23,
   "TransitionRadiation": 24,
 }
G4EmProcessSubType_inv = dict(zip(
    G4EmProcessSubType.values(), G4EmProcessSubType.keys()
    ))
