#!/usr/bin/env python3
"""
Analyze and compare neural network sizes for different configurations.

Calculates the number of parameters for each network size preset.
"""

import torch
import torch.nn as nn

# Input sizes for Flop5Holdem (from analysis)
PRIV_OBS_SIZE = 34  # 2 cards × (13 rank + 4 suit) = 34 one-hot
BOARD_SIZE = 85     # 5 cards × (13 rank + 4 suit) = 85 one-hot
PUB_OBS_SIZE = 139  # Total public observation size
HIST_SIZE = PUB_OBS_SIZE - BOARD_SIZE  # 139 - 85 = 54
N_ACTIONS = 3       # FOLD, CHECK/CALL, RAISE

# Network size presets
nn_size_presets = {
    "small": {
        "n_cards_state_units_adv": 96,
        "n_merge_and_table_layer_units_adv": 32,
        "n_units_final_adv": 32,
        "n_cards_state_units_avrg": 96,
        "n_merge_and_table_layer_units_avrg": 32,
        "n_units_final_avrg": 32,
    },
    "medium": {
        "n_cards_state_units_adv": 192,
        "n_merge_and_table_layer_units_adv": 64,
        "n_units_final_adv": 64,
        "n_cards_state_units_avrg": 192,
        "n_merge_and_table_layer_units_avrg": 64,
        "n_units_final_avrg": 64,
    },
    "large": {
        "n_cards_state_units_adv": 384,
        "n_merge_and_table_layer_units_adv": 128,
        "n_units_final_adv": 128,
        "n_cards_state_units_avrg": 384,
        "n_merge_and_table_layer_units_avrg": 128,
        "n_units_final_avrg": 128,
    },
}


def count_parameters(model):
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_mpm_flat(card_block_units, other_units):
    """Create MainPokerModuleFLAT structure (without actual forward pass)."""
    layers = {}
    
    # Private cards branch
    layers['_priv_cards'] = nn.Linear(PRIV_OBS_SIZE, other_units)
    
    # Board cards branch
    layers['_board_cards'] = nn.Linear(BOARD_SIZE, other_units)
    
    # Cards processing
    layers['cards_fc_1'] = nn.Linear(2 * other_units, card_block_units)
    layers['cards_fc_2'] = nn.Linear(card_block_units, card_block_units)
    layers['cards_fc_3'] = nn.Linear(card_block_units, other_units)
    
    # History and state processing
    layers['hist_and_state_1'] = nn.Linear(HIST_SIZE, other_units)
    layers['hist_and_state_2'] = nn.Linear(other_units, other_units)
    
    # Final layers
    layers['final_fc_1'] = nn.Linear(2 * other_units, other_units)
    layers['final_fc_2'] = nn.Linear(other_units, other_units)
    
    return layers


def create_dueling_qnet(mpm_output_units, n_units_final):
    """Create DuelingQNet structure."""
    layers = {}
    layers['_adv_layer'] = nn.Linear(mpm_output_units, n_units_final)
    layers['_state_v_layer'] = nn.Linear(mpm_output_units, n_units_final)
    layers['_adv'] = nn.Linear(n_units_final, N_ACTIONS)
    layers['_v'] = nn.Linear(n_units_final, 1)
    return layers


def calculate_network_params(card_block_units, other_units, n_units_final):
    """Calculate total parameters for advantage network."""
    mpm_layers = create_mpm_flat(card_block_units, other_units)
    qnet_layers = create_dueling_qnet(other_units, n_units_final)
    
    # Count MPM parameters
    mpm_params = sum(count_parameters(layer) for layer in mpm_layers.values())
    
    # Count DuelingQNet parameters
    qnet_params = sum(count_parameters(layer) for layer in qnet_layers.values())
    
    return mpm_params + qnet_params


def format_number(num):
    """Format number with K/M suffix."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    else:
        return str(num)


def print_network_comparison():
    """Print a formatted comparison of network sizes."""
    print("=" * 100)
    print("Neural Network Architecture Comparison for Flop5Holdem")
    print("=" * 100)
    print()
    
    # Table header
    print(f"{'Size':<10} {'Card Block':<12} {'Other Units':<12} {'Final Units':<12} {'Adv Params':<15} {'Avrg Params':<15} {'Total':<15}")
    print("-" * 100)
    
    results = []
    for size_name, config in nn_size_presets.items():
        adv_params = calculate_network_params(
            config["n_cards_state_units_adv"],
            config["n_merge_and_table_layer_units_adv"],
            config["n_units_final_adv"]
        )
        avrg_params = calculate_network_params(
            config["n_cards_state_units_avrg"],
            config["n_merge_and_table_layer_units_avrg"],
            config["n_units_final_avrg"]
        )
        total_params = adv_params + avrg_params
        
        results.append({
            'size': size_name,
            'card_block': config["n_cards_state_units_adv"],
            'other_units': config["n_merge_and_table_layer_units_adv"],
            'final_units': config["n_units_final_adv"],
            'adv_params': adv_params,
            'avrg_params': avrg_params,
            'total': total_params
        })
        
        print(f"{size_name:<10} {config['n_cards_state_units_adv']:<12} "
              f"{config['n_merge_and_table_layer_units_adv']:<12} "
              f"{config['n_units_final_adv']:<12} "
              f"{format_number(adv_params):<15} "
              f"{format_number(avrg_params):<15} "
              f"{format_number(total_params):<15}")
    
    print()
    print("=" * 100)
    print("Architecture Details")
    print("=" * 100)
    print()
    print("MainPokerModuleFLAT (with use_pre_layers=True):")
    print("  Input sizes:")
    print(f"    - Private observation: {PRIV_OBS_SIZE} (2 cards × [13 rank + 4 suit])")
    print(f"    - Board cards: {BOARD_SIZE} (5 cards × [13 rank + 4 suit])")
    print(f"    - History/state: {HIST_SIZE} (pub_obs_size - board_size)")
    print()
    print("  Layer structure:")
    print("    1. Private cards: Linear(34 → other_units)")
    print("    2. Board cards: Linear(85 → other_units)")
    print("    3. Cards FC 1: Linear(2×other_units → card_block_units)")
    print("    4. Cards FC 2: Linear(card_block_units → card_block_units) [with skip connection]")
    print("    5. Cards FC 3: Linear(card_block_units → other_units)")
    print("    6. Hist/state FC 1: Linear(54 → other_units)")
    print("    7. Hist/state FC 2: Linear(other_units → other_units) [with skip connection]")
    print("    8. Final FC 1: Linear(2×other_units → other_units)")
    print("    9. Final FC 2: Linear(other_units → other_units) [with skip connection]")
    print("    Output: other_units")
    print()
    print("DuelingQNet:")
    print("    1. Advantage layer: Linear(mpm_output → n_units_final)")
    print("    2. State value layer: Linear(mpm_output → n_units_final)")
    print("    3. Advantage head: Linear(n_units_final → 3 actions)")
    print("    4. Value head: Linear(n_units_final → 1)")
    print()
    print("=" * 100)
    print("Detailed Parameter Breakdown")
    print("=" * 100)
    print()
    
    for result in results:
        print(f"{result['size'].upper()} Configuration:")
        print(f"  Card block units: {result['card_block']}")
        print(f"  Other units: {result['other_units']}")
        print(f"  Final units: {result['final_units']}")
        print()
        
        # Calculate MPM params
        mpm_layers = create_mpm_flat(result['card_block'], result['other_units'])
        mpm_params = sum(count_parameters(layer) for layer in mpm_layers.values())
        
        # Calculate QNet params
        qnet_layers = create_dueling_qnet(result['other_units'], result['final_units'])
        qnet_params = sum(count_parameters(layer) for layer in qnet_layers.values())
        
        print(f"  Advantage Network:")
        print(f"    MainPokerModule: {format_number(mpm_params)} parameters")
        print(f"    DuelingQNet: {format_number(qnet_params)} parameters")
        print(f"    Total Adv: {format_number(result['adv_params'])} parameters")
        print()
        
        # Average network (same structure)
        avrg_mpm_layers = create_mpm_flat(result['card_block'], result['other_units'])
        avrg_mpm_params = sum(count_parameters(layer) for layer in avrg_mpm_layers.values())
        avrg_qnet_layers = create_dueling_qnet(result['other_units'], result['final_units'])
        avrg_qnet_params = sum(count_parameters(layer) for layer in avrg_qnet_layers.values())
        
        print(f"  Average Network:")
        print(f"    MainPokerModule: {format_number(avrg_mpm_params)} parameters")
        print(f"    DuelingQNet: {format_number(avrg_qnet_params)} parameters")
        print(f"    Total Avrg: {format_number(result['avrg_params'])} parameters")
        print()
        print(f"  Combined Total: {format_number(result['total'])} parameters")
        print()
        print("-" * 100)
        print()


if __name__ == "__main__":
    print_network_comparison()
