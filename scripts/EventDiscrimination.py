import matplotlib.pyplot as plt
import uproot, yaml, sys, os
import seaborn as sns
import mplhep as hep
import awkward as ak
import pandas as pd
import numpy as np

from hep_utils import invariantMass2, transverseMass
from termcolor import colored
from dotmap import DotMap
from numba import types
from tqdm import tqdm

class EventDiscrimination:

	def __init__(self, root_filepath, name):

		self.limit_events = None

		self.name = name
		self.input_path = root_filepath
		self.output_path = '../data/outputs/'

		self.count_bf = np.zeros(8)

		self.load_config('../config/config.yaml')
		self.initialize(self.input_path)
		self.run_filter()
		self.finalize(self.output_path)

	def load_config(self, config_path):
		
		with open(config_path) as conf_file:
			self.config = DotMap(yaml.load(conf_file, Loader=yaml.Loader))
			exec(self.config['plot_style']['dark'])

	def initialize(self, input_path):

		tree = uproot.open(input_path)['Delphes']

		# Input Arrays

		EVENTS_MET = np.hstack(tree['MissingET/MissingET.MET'].arrays(library='np')['MissingET.MET'])
		EVENTS_PHI = np.hstack(tree['MissingET/MissingET.Phi'].arrays(library='np')['MissingET.Phi'])

		MUONS_PT = tree['Muon/Muon.PT'].arrays(library='np')['Muon.PT']
		MUONS_ETA = tree['Muon/Muon.Eta'].arrays(library='np')['Muon.Eta']
		MUONS_PHI = tree['Muon/Muon.Phi'].arrays(library='np')['Muon.Phi']
		MUONS_Q = tree['Muon/Muon.Charge'].arrays(library='np')['Muon.Charge']
		MUONS_IVAR = tree['Muon/Muon.IsolationVar'].arrays(library='np')['Muon.IsolationVar']

		ELECS_PT = tree['Electron/Electron.PT'].arrays(library='np')['Electron.PT']
		ELECS_ETA = tree['Electron/Electron.Eta'].arrays(library='np')['Electron.Eta']
		ELECS_PHI = tree['Electron/Electron.Phi'].arrays(library='np')['Electron.Phi']
		ELECS_Q = tree['Electron/Electron.Charge'].arrays(library='np')['Electron.Charge']
		ELECS_IVAR = tree['Electron/Electron.IsolationVar'].arrays(library='np')['Electron.IsolationVar']

		# Generating Iterable for Events

		self.INPUT_PARAMS = list(enumerate(zip(
			    MUONS_PT, MUONS_ETA, MUONS_PHI, MUONS_Q, MUONS_IVAR,
			    ELECS_PT, ELECS_ETA, ELECS_PHI, ELECS_Q, ELECS_IVAR,
			    EVENTS_MET, EVENTS_PHI
		    )))

		# Output Arrays

		self.W_LEPTON_PT = np.array([], dtype=np.float32)
		self.W_LEPTON_ETA = np.array([], dtype=np.float32)
		self.Z_LEPTON_PT = np.array([], dtype=np.float32)
		self.Z_LEPTON_ETA = np.array([], dtype=np.float32)
		self.FILTERED_MET = np.array([], dtype=np.float32)
		self.LEAD_LEPTON_PT = np.array([], dtype=np.float32)
		self.SLEAD_LEPTON_PT = np.array([], dtype=np.float32)
		self.SSLEAD_LEPTON_PT = np.array([], dtype=np.float32)
		self.TRANSVERSE_MASS_W = np.array([], dtype=np.float32)
		self.INVARIANT_MASS_Z = np.array([], dtype=np.float32)
		self.TOTAL_ENERGY = np.array([], dtype=np.float32)

		# Event Selection Criteria
		
		self.min_lep_leading_pt = 25
		self.min_lep_sleading_pt = 20
		self.min_lep_ssleading_pt = 10
		self.min_ETmiss = 100
		self.max_isolation = 0.2

	def run_filter(self):
		
		self.eff_events = 0

		for event_idx, params in tqdm(self.INPUT_PARAMS, ascii=True):

			df_mPT = pd.DataFrame({
			    'PT': params[0],
			    'Eta': params[1],
			    'Phi': params[2],
			    'Charge': params[3],
			    'IsolationVar': params[4],
			    'Flavor': 'μ',
			    'Mass': self.config.mass.muon})

			df_ePT = pd.DataFrame({
			    'PT': params[5],
			    'Eta': params[6],
			    'Phi': params[7],
			    'Charge': params[8],
			    'IsolationVar': params[9],
			    'Flavor': 'e',
			    'Mass': self.config.mass.elec})

			lep = pd.concat([df_mPT, df_ePT],
			    ignore_index=True).sort_values('PT',
			        ascending=False).reset_index(drop=True)

			# 3 Exact Leptons
			if len(lep) != 3: continue

			# SFOS Criteria (Mismo sabor carga opuesta)
			SFOS_leps = []
			for idx in range(3):
			    if lep['Flavor'].iloc[idx] == lep['Flavor'].iloc[idx-1]: # Same Flavor
			        if lep['Charge'].iloc[idx] != lep['Charge'].iloc[idx-1]: # Opposite Charge Sign
			            SFOS_leps.append(idx)

			if len(SFOS_leps) == 0: continue
			    
			elif len(SFOS_leps) == 1:

				idx_lW = [1, 2, 0][SFOS_leps[0]]
				idx_lZ = (SFOS_leps[0], SFOS_leps[0]-1)

			elif len(SFOS_leps) == 2:
				
				inv_masses = []
				for idx in range(3):
					mll = invariantMass2(
						lep['PT'][idx], lep['Eta'][idx], lep['Phi'][idx], lep['Mass'][idx],
						lep['PT'].to_numpy()[idx-1], lep['Eta'].to_numpy()[idx-1], lep['Phi'].to_numpy()[idx-1], lep['Mass'].to_numpy()[idx-1])
					inv_masses = np.append(inv_masses, mll)
				idx_closest = np.argmin(np.abs(inv_masses - self.config.mass.Zboson))
				idx_lZ = [(0, 2), (1, 0), (2, 1)][idx_closest]
				idx_lW = [1, 2, 0][idx_closest]

			# Isolated Leptons
			if np.all(lep['IsolationVar'] > self.max_isolation): continue

			# Minimum of event Missing ET
			if params[10] < self.min_ETmiss: continue

			# Minimum of PT for each Lepton
			if lep['PT'][0] < self.min_lep_leading_pt: continue
			if lep['PT'][1] < self.min_lep_sleading_pt: continue
			if lep['PT'][2] < self.min_lep_ssleading_pt: continue
			
			# Getting the Transverse Mass (W)
			dphi =  params[11] - lep['Phi'][idx_lW]
			mT = transverseMass(lep['PT'][idx_lW], params[10], dphi)

			leptons = lz1, lz2, lw = lep.iloc[idx_lZ[0]], lep.iloc[idx_lZ[1]], lep.iloc[idx_lW]

			Et = 0
			for l in leptons:
				Et += l['Mass']**2 + l['PT']**2 * (1 + np.sinh(l['Eta'])**2)

			if Et < 0: Et = 0

			# Saving the results in arrays
			self.W_LEPTON_PT = np.append(self.W_LEPTON_PT, lep['PT'][idx_lW])
			self.W_LEPTON_ETA = np.append(self.W_LEPTON_ETA, lep['Eta'][idx_lW])
			self.Z_LEPTON_PT = np.append(self.Z_LEPTON_PT, (lep['PT'].to_numpy()[idx_lZ[0]], lep['PT'].to_numpy()[idx_lZ[1]]) )
			self.Z_LEPTON_ETA = np.append(self.Z_LEPTON_ETA, (lep['Eta'].to_numpy()[idx_lZ[0]], lep['Eta'].to_numpy()[idx_lZ[1]]) )
			self.FILTERED_MET = np.append(self.FILTERED_MET, params[10])
			self.LEAD_LEPTON_PT = np.append(self.LEAD_LEPTON_PT, lep['PT'][0])
			self.SLEAD_LEPTON_PT = np.append(self.SLEAD_LEPTON_PT, lep['PT'][1])
			self.SSLEAD_LEPTON_PT = np.append(self.SSLEAD_LEPTON_PT, lep['PT'][2])
			self.TRANSVERSE_MASS_W = np.append(self.TRANSVERSE_MASS_W, mT)
			self.INVARIANT_MASS_Z = np.append(self.INVARIANT_MASS_Z, inv_masses[idx_closest])
			self.TOTAL_ENERGY = np.append(self.TOTAL_ENERGY, np.sqrt(Et))

			self.eff_events += 1
			
			if self.limit_events is not None and self.eff_events == self.limit_events: break

			if lw['Flavor'] == 'e':
				if lw['Charge'] == -1:
					if lz1['Flavor'] == 'e' and lz2['Flavor'] == 'e':
						self.count_bf[0] += 1
					elif lz1['Flavor'] == 'μ' and lz2['Flavor'] == 'μ':
						self.count_bf[6] += 1
					else: print('Error')

				elif lw['Charge'] == 1:
					if lz1['Flavor'] == 'e' and lz2['Flavor'] == 'e':
						self.count_bf[2] += 1
					elif lz1['Flavor'] == 'μ' and lz2['Flavor'] == 'μ':
						self.count_bf[7] += 1
					else: print('Error')

			elif lw['Flavor'] == 'μ':
				if lw['Charge'] == -1:
					if lz1['Flavor'] == 'e' and lz2['Flavor'] == 'e':
						self.count_bf[4] += 1
					elif lz1['Flavor'] == 'μ' and lz2['Flavor'] == 'μ':
						self.count_bf[1] += 1
					else: print('Error')

				elif lw['Charge'] == 1:
					if lz1['Flavor'] == 'e' and lz2['Flavor'] == 'e':
						self.count_bf[5] += 1
					elif lz1['Flavor'] == 'μ' and lz2['Flavor'] == 'μ':
						self.count_bf[3] += 1
					else: print('Error')

	def finalize(self, output_path):
		
		np.save(output_path + f'W_LEPTON_PT_{self.name}.npy', self.W_LEPTON_PT)
		np.save(output_path + f'W_LEPTON_ETA_{self.name}.npy', self.W_LEPTON_PT)
		np.save(output_path + f'Z_LEPTON_PT_{self.name}.npy', self.W_LEPTON_PT)
		np.save(output_path + f'Z_LEPTON_ETA_{self.name}.npy', self.W_LEPTON_PT)
		np.save(output_path + f'FILTERED_MET_{self.name}.npy', self.FILTERED_MET)
		np.save(output_path + f'LEAD_LEPTON_PT_{self.name}.npy', self.LEAD_LEPTON_PT)
		np.save(output_path + f'SLEAD_LEPTON_PT_{self.name}.npy', self.SLEAD_LEPTON_PT)
		np.save(output_path + f'SSLEAD_LEPTON_PT_{self.name}.npy', self.SSLEAD_LEPTON_PT)
		np.save(output_path + f'TRANSVERSE_MASS_W_{self.name}.npy', self.TRANSVERSE_MASS_W)
		np.save(output_path + f'INVARIANT_MASS_Z_{self.name}.npy', self.INVARIANT_MASS_Z)
		np.save(output_path + f'TOTAL_ENERGY_{self.name}.npy', self.TOTAL_ENERGY)

if __name__ == '__main__':
	
	print(f'{colored('[INFO]', 'yellow')}: Running signal event discrimination')
	Signal_Analysis = EventDiscrimination('../data_profe/signal.root', 'SIGNAL') # ✔️
	print(f'{colored('[INFO]', 'green')}: Signal event discrimination completed successfully')


	print(f'{colored('[INFO]', 'yellow')}: Running bkg ZZ event discrimination')
	Bkg_ZZ_Analysis = EventDiscrimination('../data_profe/bkg_ZZ.root', 'BKG_ZZ') # ✔️
	print(f'{colored('[INFO]', 'green')}: Bkg ZZ event discrimination completed successfully')
	
	print(f'{colored('[INFO]', 'yellow')}: Running bkg WZ event discrimination')
	Bkg_WZ_Analysis = EventDiscrimination('../data_profe/bkg_WZ.root', 'BKG_WZ') # ✔️
	print(f'{colored('[INFO]', 'green')}: Bkg WZ event discrimination completed successfully')
	
	print(f'{colored('[INFO]', 'yellow')}: Running bkg WW event discrimination')
	Bkg_WW_Analysis = EventDiscrimination('../data_profe/bkg_WW.root', 'BKG_WW') # ✔️
	print(f'{colored('[INFO]', 'green')}: Bkg WW event discrimination completed successfully')

	print(f'{colored('[INFO]', 'yellow')}: Running bkg tt event discrimination')
	Bkg_tt_Analysis = EventDiscrimination('../data_profe/bkg_tt.root', 'BKG_tt') # ✔️
	print(f'{colored('[INFO]', 'green')}: Bkg tt event discrimination completed successfully')