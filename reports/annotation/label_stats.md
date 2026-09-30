# Label statistics: `annotations/labels_v1.json`

Clips: 200; complete: 200; view_usable: 195; warnings: 0

## Events per test type and role (all tags)

config                      ITV               VTB               VTP         
role                   exterior occupant exterior occupant exterior occupant
event                                                                       
airbag_deploy_start           0       29       39       40        0       29
head_contact                  0       25        0       38        0       23
impact_onset                 29       30       40       40       30       30
max_deformation              29        0       39        0       29        0
side_glass_break_onset       12       11        0        0       18        9
windshield_crack_onset        0        0        0        0       14       27

## Onset timing, ms (visible onsets on time-verified clips)

                                        count   25%   50%   75%
event                  config role                             
airbag_deploy_start    ITV    occupant   29.0  14.0  14.0  14.4
                       VTB    occupant   39.0  22.0  22.0  22.2
                       VTP    occupant   25.0  14.0  14.0  14.9
head_contact           ITV    occupant   25.0  38.0  46.0  49.5
                       VTB    occupant   37.0  46.0  46.0  46.0
                       VTP    occupant   23.0  38.0  38.1  46.0
max_deformation        ITV    exterior   29.0  62.0  62.2  70.0
                       VTB    exterior   36.0  70.0  70.0  70.0
                       VTP    exterior   28.0  70.5  78.0  78.0
side_glass_break_onset ITV    exterior   12.0  30.0  38.1  45.2
                              occupant   11.0  22.0  22.0  25.9
                       VTP    exterior   17.0  38.0  39.0  46.0
                              occupant    9.0  14.0  29.0  38.0
windshield_crack_onset VTP    exterior   14.0  39.8  46.0  46.0
                              occupant   26.0  38.0  46.0  46.0

## Clip summary attributes

- frontal_airbag_deployed: {'yes': 90, 'not_visible': 66, 'no': 44}
- curtain_airbag_deployed: {'yes': 88, 'not_visible': 70, 'no': 42}
- door_opened: {'no': 120, 'not_visible': 78, 'yes': 2}
- fire_or_fluid_leak: {'no': 200}
- visual_severity: {'medium': 125, 'high': 75}
