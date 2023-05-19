import numpy as np
from gym import spaces

vnfs_size = 20

space_dictionary = {'nb_resources': spaces.Box(low=0, high=vnfs_size, shape=(1,), dtype=np.uint8),
                    'id': spaces.Box(low=0, high=100, shape=(vnfs_size,1), dtype=np.uint8),
                    'state': spaces.Box(low=0, high=2, shape=(vnfs_size,1), dtype=np.uint8),
                    'attack_type': spaces.Box(low=0, high=5, shape=(vnfs_size,1), dtype=np.uint8),
                    'vuln_ports_count': spaces.Box(low=0, high=100, shape=(vnfs_size,1), dtype=np.uint8),
                    'apt_scores': spaces.Box(low=0, high=1000000, shape=(vnfs_size,8), dtype=np.float32),
                    'data_leak_scores': spaces.Box(low=-1000000, high=1000000, shape=(vnfs_size,8), dtype=np.float32),
                    'dos_scores': spaces.Box(low=-1000000, high=1000000, shape=(vnfs_size,8), dtype=np.float32),
                    'undefined_scores': spaces.Box(low=-1000000, high=1000000, shape=(vnfs_size,8), dtype=np.float32),
                    'resource_consumption': spaces.Box(low=0, high=1000000, shape=(vnfs_size,3), dtype=np.float32),
                    'nb_UEs_cnx': spaces.Box(low=0, high=255, shape=(vnfs_size,1), dtype=np.uint8),
                    'vim_host':spaces.Box(low=0, high=2, shape=(vnfs_size,1), dtype=np.uint8),
                    'vim_resources': spaces.Box(low=0, high=999999999, shape=(vnfs_size,3), dtype=np.float64),
                    'location': spaces.Box(low=0, high=1, shape=(vnfs_size,1), dtype=np.uint8),
                    'network_metrics': spaces.Box(low=-1, high=999999999, shape=(vnfs_size,5), dtype=np.float64),
                    'latency_sla': spaces.Box(low=0, high=1, shape=(vnfs_size,1), dtype=np.float32),
                    'impact_ssla': spaces.Box(low=0, high=3, shape=(vnfs_size,1), dtype=np.uint8),
                    'vnf_parent': spaces.Box(low=0, high=vnfs_size, shape=(vnfs_size,1), dtype=np.uint8),
                    'ns_parents': spaces.Box(low=0, high=100, shape=(vnfs_size,4), dtype=np.uint8),
                    'nsi_parents': spaces.Box(low=0, high=100, shape=(vnfs_size,4), dtype=np.uint8),
                    'mtd_action': spaces.Box(low=-20, high=255, shape=(vnfs_size,2), dtype=np.float16), # nothing, restart, or migrate + duration of the action in seconds
                    'mtd_constraint': spaces.Box(low=0, high=1000, shape=(vnfs_size,2), dtype=np.uint8)} # remaining migrations + remaining reinst.


space_set_zeros = {'nb_resources': np.zeros(shape=(1,), dtype=np.uint8),
            'id': np.zeros(shape=(vnfs_size,1), dtype=np.uint8),
            'state': np.zeros(shape=(vnfs_size,1), dtype=np.uint8),
            'attack_type': np.zeros(shape=(vnfs_size,1), dtype=np.uint8),
            'vuln_ports_count': np.zeros(shape=(vnfs_size,1), dtype=np.uint8),
            'apt_scores': np.zeros(shape=(vnfs_size,8), dtype=np.float32),
            'data_leak_scores': np.zeros(shape=(vnfs_size,8), dtype=np.float32),
            'dos_scores': np.zeros(shape=(vnfs_size,8), dtype=np.float32),
            'undefined_scores': np.zeros(shape=(vnfs_size,8), dtype=np.float32),
            'resource_consumption': np.zeros(shape=(vnfs_size,3), dtype=np.float32),
            'nb_UEs_cnx': np.zeros(shape=(vnfs_size,1), dtype=np.uint8),
            'vim_host':np.zeros(shape=(vnfs_size,1), dtype=np.uint8),
            'vim_resources': np.zeros(shape=(vnfs_size, 3), dtype=np.float64),
            'location': np.zeros(shape=(vnfs_size,1), dtype=np.uint8),
            'network_metrics': np.zeros(shape=(vnfs_size,5), dtype=np.float64),
            'latency_sla': np.zeros(shape=(vnfs_size,1), dtype=np.float32),
            'impact_ssla': np.zeros(shape=(vnfs_size,1), dtype=np.uint8),
            'vnf_parent': np.zeros(shape=(vnfs_size,1), dtype=np.uint8),
            'ns_parents': np.zeros(shape=(vnfs_size,4), dtype=np.uint8),
            'nsi_parents': np.zeros(shape=(vnfs_size,4), dtype=np.uint8),
            'mtd_action': np.zeros(shape=(vnfs_size,2), dtype=np.float16),
            'mtd_constraint': np.zeros(shape=(vnfs_size,2), dtype=np.uint8)}


reward_init = {'resource_reward':0,
               'network_reward': 0,
               'proactive_security_reward':0}

migrations_per_month = 60
reinstantiations_per_month = 330

# STARTING NETWORK SETUP
locations = {'core':0, 'edge':1}
# vnf0 has strict latency_ssla
vnf0 = {'id': 4, 'state': 'ordinary', 'attack_type': None, 'apt_cvss_score_min': 0, 'vuln_ports_count': 1, 'apt_cvss_score_max': 0, 'apt_cvss_score_avg': 0, 'apt_cvss_score_std': 0, 'apt_cvss_asp_min': 0, 'apt_cvss_asp_max': 0, 'apt_cvss_asp_avg': 0, 'apt_cvss_asp_std': 0, 'data_leak_cvss_score_min': 4.4, 'data_leak_cvss_score_max': 4.4, 'data_leak_cvss_score_avg': 4.4, 'data_leak_cvss_score_std': 0.0, 'data_leak_cvss_asp_min': 2.9589945, 'data_leak_cvss_asp_max': 2.9589945, 'data_leak_cvss_asp_avg': 2.9589945, 'data_leak_cvss_asp_std': 0.0, 'dos_cvss_score_min': 0, 'dos_cvss_score_max': 0, 'dos_cvss_score_avg': 0, 'dos_cvss_score_std': 0, 'dos_cvss_asp_min': 0, 'dos_cvss_asp_max': 0, 'dos_cvss_asp_avg': 0, 'dos_cvss_asp_std': 0, 'undefined_cvss_score_min': 0, 'undefined_cvss_score_max': 0, 'undefined_cvss_score_avg': 0, 'undefined_cvss_score_std': 0, 'undefined_cvss_asp_min': 0, 'undefined_cvss_asp_max': 0, 'undefined_cvss_asp_avg': 0, 'undefined_cvss_asp_std': 0, 'vim_host': 'VIM 2', 'vim_cpus': 16, 'vim_ram_mb': 96666, 'vim_disk_gb': 2013, 'vim_location': 'edge', 'cpu_cons': 2, 'ram_cons': 2048, 'disk_cons': 10, 'nb_UEs_cnx': None, 'latency': 0, 'throughput': 0, 'packet_loss_rate': 0, 'nb_pck_out': 0, 'nb_pck_in': 0, 'latency_sla': 0.05, 'impact_ssla':3, 'vnf_parent': 'VNF 3', 'ns_parent1': 'NS 2', 'ns_parent2': None, 'ns_parent3': None, 'ns_parent4': None, 'nsi_parent1': 'NSi 1', 'nsi_parent2': None, 'nsi_parent3': None, 'nsi_parent4': None, 'network_overhead_cumul_avg': 0.0, 'incremental_counter': 1}
# vnf1 has great impact_ssla
vnf1 = {'id': 7, 'state': 'ordinary', 'attack_type': None, 'apt_cvss_score_min': 0, 'vuln_ports_count': 0, 'apt_cvss_score_max': 0, 'apt_cvss_score_avg': 0, 'apt_cvss_score_std': 0, 'apt_cvss_asp_min': 0, 'apt_cvss_asp_max': 0, 'apt_cvss_asp_avg': 0, 'apt_cvss_asp_std': 0, 'data_leak_cvss_score_min': 0, 'data_leak_cvss_score_max': 0, 'data_leak_cvss_score_avg': 0, 'data_leak_cvss_score_std': 0, 'data_leak_cvss_asp_min': 0, 'data_leak_cvss_asp_max': 0, 'data_leak_cvss_asp_avg': 0, 'data_leak_cvss_asp_std': 0, 'dos_cvss_score_min': 0, 'dos_cvss_score_max': 0, 'dos_cvss_score_avg': 0, 'dos_cvss_score_std': 0, 'dos_cvss_asp_min': 0, 'dos_cvss_asp_max': 0, 'dos_cvss_asp_avg': 0, 'dos_cvss_asp_std': 0, 'undefined_cvss_score_min': 0, 'undefined_cvss_score_max': 0, 'undefined_cvss_score_avg': 0, 'undefined_cvss_score_std': 0, 'undefined_cvss_asp_min': 0, 'undefined_cvss_asp_max': 0, 'undefined_cvss_asp_avg': 0, 'undefined_cvss_asp_std': 0, 'vim_host': 'VIM 2', 'vim_cpus': 16, 'vim_ram_mb': 96666, 'vim_disk_gb': 2013, 'vim_location': 'edge', 'cpu_cons': 2, 'ram_cons': 2048, 'disk_cons': 10, 'nb_UEs_cnx': None, 'latency': 0, 'throughput': 0, 'packet_loss_rate': 0, 'nb_pck_out': 0, 'nb_pck_in': 0, 'latency_sla': 0.05, 'impact_ssla':0, 'vnf_parent': 'VNF 6', 'ns_parent1': 'NS 5', 'ns_parent2': None, 'ns_parent3': None, 'ns_parent4': None, 'nsi_parent1': 'NSi 1', 'nsi_parent2': None, 'nsi_parent3': None, 'nsi_parent4': None, 'network_overhead_cumul_avg': 0.0, 'incremental_counter': 1}
vnf2 = {'id': 13, 'state': 'ordinary', 'attack_type': None, 'apt_cvss_score_min': 0, 'vuln_ports_count': 0, 'apt_cvss_score_max': 0, 'apt_cvss_score_avg': 0, 'apt_cvss_score_std': 0, 'apt_cvss_asp_min': 0, 'apt_cvss_asp_max': 0, 'apt_cvss_asp_avg': 0, 'apt_cvss_asp_std': 0, 'data_leak_cvss_score_min': 0, 'data_leak_cvss_score_max': 0, 'data_leak_cvss_score_avg': 0, 'data_leak_cvss_score_std': 0, 'data_leak_cvss_asp_min': 0, 'data_leak_cvss_asp_max': 0, 'data_leak_cvss_asp_avg': 0, 'data_leak_cvss_asp_std': 0, 'dos_cvss_score_min': 0, 'dos_cvss_score_max': 0, 'dos_cvss_score_avg': 0, 'dos_cvss_score_std': 0, 'dos_cvss_asp_min': 0, 'dos_cvss_asp_max': 0, 'dos_cvss_asp_avg': 0, 'dos_cvss_asp_std': 0, 'undefined_cvss_score_min': 0, 'undefined_cvss_score_max': 0, 'undefined_cvss_score_avg': 0, 'undefined_cvss_score_std': 0, 'undefined_cvss_asp_min': 0, 'undefined_cvss_asp_max': 0, 'undefined_cvss_asp_avg': 0, 'undefined_cvss_asp_std': 0, 'vim_host': 'VIM 2', 'vim_cpus': 16, 'vim_ram_mb': 96666, 'vim_disk_gb': 2013, 'vim_location': 'edge', 'cpu_cons': 2, 'ram_cons': 2048, 'disk_cons': 10, 'nb_UEs_cnx': None, 'latency': 0, 'throughput': 0, 'packet_loss_rate': 0, 'nb_pck_out': 0, 'nb_pck_in': 0, 'latency_sla': 0.05, 'impact_ssla':0, 'vnf_parent': 'VNF 9', 'ns_parent1': 'NS 8', 'ns_parent2': None, 'ns_parent3': None, 'ns_parent4': None, 'nsi_parent1': 'NSi 1', 'nsi_parent2': None, 'nsi_parent3': None, 'nsi_parent4': None, 'network_overhead_cumul_avg': 0.0, 'incremental_counter': 1}
# vnf3 has apt vulnerabilities
vnf3 = {'id': 10, 'state': 'ordinary', 'attack_type': None, 'apt_cvss_score_min': 2, 'vuln_ports_count': 2, 'apt_cvss_score_max': 6.5, 'apt_cvss_score_avg': 5.5, 'apt_cvss_score_std': 2, 'apt_cvss_asp_min': 4, 'apt_cvss_asp_max': 8, 'apt_cvss_asp_avg': 6.5, 'apt_cvss_asp_std': 0.6, 'data_leak_cvss_score_min': 4.3, 'data_leak_cvss_score_max': 8.8, 'data_leak_cvss_score_avg': 6.957142857142856, 'data_leak_cvss_score_std': 2.02819963267919, 'data_leak_cvss_asp_min': 2.0680681560000003, 'data_leak_cvss_asp_max': 2.9589945, 'data_leak_cvss_asp_avg': 2.6589255577499995, 'data_leak_cvss_asp_std': 0.34346145952646673, 'dos_cvss_score_min': None, 'dos_cvss_score_max': None, 'dos_cvss_score_avg': None, 'dos_cvss_score_std': None, 'dos_cvss_asp_min': None, 'dos_cvss_asp_max': None, 'dos_cvss_asp_avg': None, 'dos_cvss_asp_std': None, 'undefined_cvss_score_min': None, 'undefined_cvss_score_max': None, 'undefined_cvss_score_avg': None, 'undefined_cvss_score_std': None, 'undefined_cvss_asp_min': None, 'undefined_cvss_asp_max': None, 'undefined_cvss_asp_avg': None, 'undefined_cvss_asp_std': None, 'vim_host': 'VIM 2', 'vim_cpus': 16, 'vim_ram_mb': 96666, 'vim_disk_gb': 2013, 'vim_location': 'edge', 'cpu_cons': 4, 'ram_cons': 8096, 'disk_cons': 30, 'nb_UEs_cnx': None, 'latency': 0, 'throughput': 0, 'packet_loss_rate': 0, 'nb_pck_out': 0, 'nb_pck_in': 0, 'latency_sla': 0.05, 'impact_ssla':0, 'vnf_parent': 'VNF 12', 'ns_parent1': 'NS 11', 'ns_parent2': None, 'ns_parent3': None, 'ns_parent4': None, 'nsi_parent1': 'NSi 1', 'nsi_parent2': None, 'nsi_parent3': None, 'nsi_parent4': None, 'network_overhead_cumul_avg': 0.0, 'incremental_counter': 1}
vnfs_list = [vnf0, vnf1, vnf2, vnf3]

#convert all None in vnfs in vnfs_list in to 0
for vnf in vnfs_list:
    for key in vnf:
        if vnf[key] is None:
            vnf[key] = 0

def init_network_setup(observation):
    """" for the first observation we simulate one slice with 4 vnfs (our testbed)
         Set SLAs and resource requirements
    """
    observation['nb_resources'][0] = len(vnfs_list)
    for i, vnf in enumerate(vnfs_list):
        observation['id'][i][0] = vnf['id']
        # set variable to 2 if vnf is under attack
        if vnf['state'] == 'ordinary':
            observation['state'][i][0] = 0
        elif vnf['state'] == 'suspicious':
            observation['state'][i][0] = 1
        elif vnf['state'] == 'attack':
            observation['state'][i][0] = 2
        observation['vim_host'][i][0] = int(vnf['vim_host'].split('VIM ')[1])
        observation['location'][i][0] = locations[vnf['vim_location']]

        # added this mainly for the simulation of a vulnerable vnf
        observation['apt_scores'][i][0] = vnf['apt_cvss_score_min']
        observation['apt_scores'][i][1] = vnf['apt_cvss_score_max']
        observation['apt_scores'][i][2] = vnf['apt_cvss_score_avg']
        observation['apt_scores'][i][3] = vnf['apt_cvss_score_std']
        observation['apt_scores'][i][4] = vnf['apt_cvss_asp_min']
        observation['apt_scores'][i][5] = vnf['apt_cvss_asp_max']
        observation['apt_scores'][i][6] = vnf['apt_cvss_asp_avg']
        observation['apt_scores'][i][7] = vnf['apt_cvss_asp_std']

        observation['vim_resources'][i][0] = vnf['vim_cpus']
        observation['vim_resources'][i][1] = vnf['vim_ram_mb']
        observation['vim_resources'][i][2] = vnf['vim_disk_gb']
        observation['resource_consumption'][i][0] = vnf['cpu_cons']
        observation['resource_consumption'][i][1] = vnf['ram_cons']
        observation['resource_consumption'][i][2] = vnf['disk_cons']
        observation['vnf_parent'][i][0] = int(vnf['vnf_parent'].split('VNF ')[1])
        observation['ns_parents'][i][0] = 0 if vnf['ns_parent1'] == 0 else int(vnf['ns_parent1'].split('NS ')[1])
        observation['ns_parents'][i][1] = 0 if vnf['ns_parent2'] == 0 else int(vnf['ns_parent2'].split('NS ')[1])
        observation['ns_parents'][i][2] = 0 if vnf['ns_parent3'] == 0 else int(vnf['ns_parent3'].split('NS ')[1])
        observation['ns_parents'][i][3] = 0 if vnf['ns_parent4'] == 0 else int(vnf['ns_parent4'].split('NS ')[1])
        observation['nsi_parents'][i][0] = 0 if vnf['nsi_parent1'] == 0 else int(vnf['nsi_parent1'].split('NSi ')[1])
        observation['nsi_parents'][i][1] = 0 if vnf['nsi_parent2'] == 0 else int(vnf['nsi_parent2'].split('NSi ')[1])
        observation['nsi_parents'][i][2] = 0 if vnf['nsi_parent3'] == 0 else int(vnf['nsi_parent3'].split('NSi ')[1])
        observation['nsi_parents'][i][3] = 0 if vnf['nsi_parent4'] == 0 else int(vnf['nsi_parent4'].split('NSi ')[1])
        observation['nb_UEs_cnx'][i][0] = 4
        observation['impact_ssla'][i][0] = vnf['impact_ssla']
        if i == 0:
            observation['latency_sla'][i][0] = 0.0165 # black sheep to see if it stays in the edge
            observation['nb_UEs_cnx'][i][0] = 10
        else:
            observation['latency_sla'][i][0] = 0.05
        observation['mtd_constraint'][i] = [reinstantiations_per_month, migrations_per_month]
    return observation
            
space_init = init_network_setup(space_set_zeros)