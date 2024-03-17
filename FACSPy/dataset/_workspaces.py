import os
from typing import Optional, Union

from lxml import etree
import numpy as np
import pandas as pd
import itertools
import copy
import re

from ..gates._gml_gates import (GMLRectangleGate,
                                GMLBooleanGate,
                                GMLPolygonGate,
                                GMLQuadrantGate)
from ..gates._gates import PolygonGate, RectangleGate
from ..gates._wsp_gates import WSPEllipsoidGate
from ..gates._gate_utils import find_attribute_value
from ..gates.dimension import Dimension

from ..transforms._matrix import Matrix
from ..transforms import _transforms, _wsp_transforms

from ..exceptions._supplements import SupplementFileNotFoundError

wsp_gate_constructor_lut = {
            'RectangleGate': GMLRectangleGate,
            'PolygonGate': GMLPolygonGate,
            'EllipsoidGate': WSPEllipsoidGate,
            'QuadrantGate': GMLQuadrantGate,
            'BooleanGate': GMLBooleanGate
        }

class FlowJoWorkspace:
    #TODO: refactor self._convert_wsp_gate
    #TODO: refactor self.parse_wsp_transforms
    def __init__(self,
                 file: str,
                 ignore_transforms: bool = False) -> None:
        
        #self.resource_path = resource_filename("FACSPy", "_resources")
        self.original_filename = os.path.basename(file)
        self.ignore_transforms = ignore_transforms
        if not os.path.isfile(file):
            raise SupplementFileNotFoundError(os.path.basename(file))
        self.wsp_dict = self.parse_workspace(file)

    def __repr__(self):
        
        return (
            f"{self.__class__.__name__}(" +
            f"{len(list(self.wsp_dict.keys()))} groups: {list(self.wsp_dict.keys())}, " +
            f"{len(list(self.wsp_dict['All Samples'].keys()))} entries.)"
        )

    def add_external_compensation(self,
                                  matrix: pd.DataFrame) -> None:
        comp_matrix = Matrix(matrix_id = "user_supplied",
                             spill_data_or_file = matrix.values,
                             detectors = matrix.columns)
        for group in self.wsp_dict.keys():
            if group == "Compensation":
                continue
            for sample in self.wsp_dict[group].keys():
                self.wsp_dict[group][sample]["compensation"] = comp_matrix
        
        return

    def parse_workspace(self,
                        file: str) -> dict[dict]:
        (wsp_root,
         gating_namespace,
         data_type_namespace,
         transform_namespace) = self._extract_namespaces(file)

        ns_map = wsp_root.nsmap
        
        group_node_list: list[etree._Element] = self.parse_xml_group_nodes(wsp_root, ns_map)
        self.raw_wsp_groups = self.parse_wsp_groups(group_node_list,
                                                    ns_map,
                                                    gating_namespace,
                                                    data_type_namespace)
        
        sample_list: list[etree._Element] = self.parse_xml_samples(wsp_root, ns_map)
        self.raw_wsp_samples = self.parse_wsp_samples(sample_list, ns_map,
                                                      gating_namespace,
                                                      transform_namespace,
                                                      data_type_namespace)
        
        return self.create_workspace_dictionary(self.raw_wsp_groups, self.raw_wsp_samples)

    def _convert_wsp_gate(self, wsp_gate, comp_matrix, xform_lut, ignore_transforms=False):
        new_dims = []
        xforms = []

        for dim in wsp_gate.dimensions:
            dim_id = dim.id

            if comp_matrix is not None:
                pre = comp_matrix['prefix']
                suf = comp_matrix['suffix']
                if dim_id.startswith(pre):
                    dim_id = re.sub(f'^{pre}', '', dim_id)
                if dim_id.endswith(suf):
                    dim_id = re.sub(f'{suf}$', '', dim_id)

                if dim_id in comp_matrix['detectors']:
                    comp_ref = comp_matrix['matrix_name']
                else:
                    comp_ref = None
            else:
                comp_ref = None

            xform_id = None
            new_dim_min = None
            new_dim_max = None

            if dim_id in xform_lut and not ignore_transforms:
                xform = xform_lut[dim_id]
                xforms.append(xform)  # need these later for vertices, coordinates, etc.
                xform_id = xform.id
                if dim.min is not None:
                    new_dim_min = xform.apply(np.array([[float(dim.min)]]))

                if dim.max is not None:
                    new_dim_max = xform.apply(np.array([[float(dim.max)]]))
            else:
                xforms.append(None)
                if dim.min is not None:
                    new_dim_min = float(dim.min)

                if dim.max is not None:
                    new_dim_max = float(dim.max)

            new_dim = Dimension(
                dim_id,
                comp_ref,
                xform_id,
                range_min=new_dim_min,
                range_max=new_dim_max
            )
            new_dims.append(new_dim)

        if isinstance(wsp_gate, GMLPolygonGate):
            # convert vertices using saved xforms
            vertices = copy.deepcopy(wsp_gate.vertices)
            for vertex in vertices:
                for i, coordinate in enumerate(vertex):
                    if xforms[i] is not None:
                        vertex[i] = xforms[i].apply(np.array([[float(coordinate)]]))[0][0]

            gate = PolygonGate(wsp_gate.gate_name, new_dims, vertices)
        elif isinstance(wsp_gate, GMLRectangleGate):
            gate = copy.deepcopy(wsp_gate)
            gate.dimensions = new_dims
        elif isinstance(wsp_gate, WSPEllipsoidGate):
            # FlowJo ellipse gates must be converted to approximate polygons.
            # When a mixed set of transforms where 1 transform is the biex,
            # the ellipse is not a true ellipse. FlowJo also converts all
            # ellipses to polygons internally when processing gates.
            gate = wsp_gate.convert_to_polygon_gate(xforms)
            gate.dimensions = new_dims
        else:
            raise NotImplementedError(
                f"{type(wsp_gate).__name__} gates for FlowJo workspaces are not currently supported."
            )

        return gate
   
    def create_workspace_dictionary(self,
                                    raw_wsp_groups: dict,
                                    raw_wsp_samples: dict) -> dict:
        wsp_dict = {}
        for group_id, group_dict in raw_wsp_groups.items():
            wsp_dict[group_id] = {}
            for sample_id in group_dict["samples"]:
                sample_dict = raw_wsp_samples[sample_id]
                sample_name = sample_dict['sample_name']
                wsp_dict[group_id][sample_name] = self.assemble_sample_from_raw_sample(sample_dict, sample_name, group_dict)
        return wsp_dict

    def parse_group_gates(self,
                          group_dict: dict,
                          sample_dict: dict) -> tuple[list[dict], list[str]]:
        
        # check the sample's sample_gates. If it is empty, then the
        # sample belongs to a group, but it has no gate hierarchy.

#                if len(sample_dict['sample_gates']) == 0:
#                    print("skipped that...")
#                    continue     
        group_sample_gate_names = []
        group_sample_gates = []
        
        if group_dict["gates"] is None:
            return [], []
        
        for group_gate in group_dict['gates']:
            group_gate_name = group_gate['gate'].gate_name

            tmp_gate = copy.deepcopy(group_gate['gate'])

            if group_gate_name in sample_dict['custom_gate_ids']:
                group_gate_path = group_gate['gate_path']
                for sample_gate_dict in sample_dict['custom_gates']:
                    tmp_sample_gate = sample_gate_dict['gate']
                    tmp_sample_gate_path = sample_gate_dict['gate_path']
                    if group_gate_path == tmp_sample_gate_path and tmp_sample_gate.gate_name == group_gate_name:
                        # found a match, overwrite tmp_gate
                        tmp_gate = tmp_sample_gate

            tmp_gate = self._convert_wsp_gate(
                tmp_gate,
                sample_dict['comp'],
                sample_dict['transforms'],
                ignore_transforms=self.ignore_transforms
            )

            group_sample_gate_names.append(group_gate_name)
            group_sample_gates.append(
                {
                    'gate': tmp_gate,
                    'gate_path': group_gate['gate_path']
                }
            )
        
        return group_sample_gates, group_sample_gate_names


    def parse_custom_gates(self,
                           sample_dict: dict,
                           group_sample_gate_names: list[str],
                           group_sample_gates: list[dict]) -> list[dict]:
        # Now, we need to check if there were only custom sample gates
        # and no group gates. In this case the above would never have
        # found the custom sample gates, but we don't want to replicate
        # them.
        for sample_gate_dict in sample_dict['custom_gates']:
            # noinspection PyTypeChecker
            sample_gate = sample_gate_dict['gate']
            # noinspection PyUnresolvedReferences
            if sample_gate.gate_name not in group_sample_gate_names:
                # noinspection PyTypeChecker
                tmp_gate = self._convert_wsp_gate(
                    sample_gate,
                    sample_dict['comp'],
                    sample_dict['transforms'],
                    ignore_transforms=self.ignore_transforms
                )
                # noinspection PyTypeChecker
                sample_gate_path = sample_gate_dict['gate_path']

                group_sample_gates.append(
                    {
                        'gate': tmp_gate,
                        'gate_path': sample_gate_path
                    }
                )

        return group_sample_gates
    
    def parse_raw_sample_gates(self,
                               sample_dict: dict,
                               group_dict: dict
                               ) -> list[dict]:        
        
        (group_sample_gates,
         group_sample_gate_names) = self.parse_group_gates(group_dict, sample_dict)
        
        custom_gates = self.parse_custom_gates(sample_dict, group_sample_gate_names, group_sample_gates)

        return group_sample_gates + custom_gates

    def assemble_sample_from_raw_sample(self,
                                        sample_dict: dict,
                                        sample_name: str,
                                        group_dict: dict) -> dict:
        
        group_sample_gates = self.parse_raw_sample_gates(sample_dict, group_dict)
        matrix = sample_dict['comp']['matrix'] if sample_dict['comp'] is not None else None
        transforms = list(sample_dict['transforms'].values())
        return {
            'gates': group_sample_gates,
            'transforms': transforms,
            'compensation': matrix
        }

    def parse_sample_gates(self,
                           sample_node: etree._Element,
                           gating_namespace: str,
                           data_type_namespace: str,
                           ns_map: dict) -> list[dict]:
        sample_root_subpopulation: etree._Element = sample_node.find("Subpopulations", ns_map)

        if sample_root_subpopulation is None:
            return []
        else:
            sample_gates = self.parse_wsp_subpopulations(
                sample_root_subpopulation,
                None,
                gating_namespace,
                data_type_namespace
            )
        return sample_gates

    def parse_wsp_samples(self,
                          sample_list: list[etree._Element],
                          ns_map: dict,
                          gating_namespace: str,
                          transform_namespace: str,
                          data_type_namespace: str) -> dict:
        wsp_samples = {}

        for sample in sample_list:
            sample_node: etree._Element = sample.find("SampleNode", ns_map)
            sample_id = sample_node.attrib["sampleID"]
            
            sample_xform_lut = self.parse_wsp_transforms(sample, transform_namespace, data_type_namespace, ns_map)
            sample_keywords_lut = self.parse_wsp_keywords(sample, ns_map)
            sample_comp = self.parse_wsp_compensation(sample, transform_namespace, data_type_namespace)
            sample_gates = self.parse_sample_gates(sample_node, gating_namespace, data_type_namespace, ns_map)
            
            # sample gate LUT will store everything we need to convert sample gates,
            # including any custom gates (ones with empty string owning groups).
            wsp_samples[sample_id] = {
                'sample_name': sample_node.attrib["name"],
                'sample_gates': sample_gates,
                'custom_gate_ids': set(),
                'custom_gates': [],
                'transforms': sample_xform_lut,
                'keywords': sample_keywords_lut,
                'comp': sample_comp
            }

            wsp_samples = self.process_custom_gates(sample_gates, wsp_samples, sample_id)

        return wsp_samples         

    def process_custom_gates(self,
                             sample_gates: list[dict],
                             wsp_samples: dict,
                             sample_id: str) -> dict:
        for sample_gate in sample_gates:
            if sample_gate['owning_group'] == '':
                # If the owning group is an empty string, it is a custom gate for that sample
                # that is potentially used in another group. However, it appears that if a
                # sample has a custom gate then that custom gate cannot be further customized.
                # Since there is only a single custom gate per gate name per sample, then we
                # can create a LUT of custom gates per sample
                wsp_samples[sample_id]['custom_gate_ids'].add(sample_gate['gate'].gate_name)
                wsp_samples[sample_id]['custom_gates'].append(
                    {
                        'gate': sample_gate['gate'],
                        'gate_path': sample_gate['gate_path']
                    }
                )
        return wsp_samples

    def parse_detectors(self,
                        matrix_element: etree._Element,
                        data_type_ns: str) -> list[str]:
        params_els: etree._Element = matrix_element.find(
            f'{data_type_ns}:parameters', matrix_element.nsmap
        )
        param_els: list[etree._Element] = params_els.findall(
            f'{data_type_ns}:parameter', matrix_element.nsmap
        )
        return [find_attribute_value(param_el, data_type_ns, 'name') for param_el in param_els]


    def parse_matrix(self,
                     matrix_element: etree._Element,
                     transform_ns: str) -> np.ndarray:
        
        return self.assemble_matrix(matrix_element, transform_ns)
    
    def parse_matrix_row(self,
                         coefficients: list[etree._Element],
                         transform_ns: str) -> np.ndarray:
        return np.array([float(find_attribute_value(coefficient, transform_ns, "value")) for coefficient in coefficients]) 

    def parse_coefficients(self,
                           spill_element: etree._Element,
                           transform_ns: str) -> list[etree._Element]:
        return spill_element.findall(f'{transform_ns}:coefficient', spill_element.nsmap)

    def assemble_matrix(self,
                        matrix_element: etree._Element,
                        transform_ns: str):
        spill_els: list[etree._Element] = matrix_element.findall(
            f'{transform_ns}:spillover', matrix_element.nsmap
        )
        return np.array([self.parse_matrix_row(self.parse_coefficients(spill_el, transform_ns), transform_ns) for spill_el in spill_els])

    def parse_wsp_compensation(self,
                               sample: etree._Element,
                               transform_ns: str,
                               data_type_ns: str) -> Optional[dict]:
        
        matrix_elements = sample.findall(f'{transform_ns}:spilloverMatrix', sample.nsmap)

        if len(matrix_elements) > 1:
            raise ValueError("Multiple spillover matrices per sample are not supported.")
        if not matrix_elements:
            return None

        matrix_element: etree._Element = matrix_elements[0]

        detectors = self.parse_detectors(matrix_element, data_type_ns)
        matrix_array = self.parse_matrix(matrix_element, transform_ns)

        matrix = Matrix(
            matrix_id = matrix_element.attrib['name'],
            spill_data_or_file = matrix_array,
            detectors=detectors,
            fluorochromes=['' for _ in detectors]
        )

        return {
            'matrix_name': matrix_element.attrib['name'],
            'prefix': matrix_element.attrib['prefix'],
            'suffix': matrix_element.attrib['suffix'],
            'detectors': detectors,
            'matrix_array': matrix_array,
            'matrix': matrix,
        }


    def parse_wsp_keywords(self,
                           sample: etree._Element,
                           ns_map: dict) -> dict:
        keywords: etree._Element = sample.find("Keywords", ns_map)
        keyword_els = keywords.getchildren()

        return {
            keyword_el.attrib['name']: keyword_el.attrib['value']
            for keyword_el in keyword_els
        }

    def parse_wsp_transforms(self,
                             sample: etree._Element,
                             transform_ns: str,
                             data_type_ns: str,
                             ns_map: dict) -> dict:
        transforms_el: etree._Element = sample.find("Transformations", ns_map)
        xform_els: list[etree._Element] = transforms_el.getchildren()

        # there should be one transform per channel, use the channel names to create a LUT
        xforms_lut = {}

        for xform_el in xform_els:
            xform_type = xform_el.tag.partition('}')[-1]

            param_el = xform_el.find(f'{data_type_ns}:parameter', xform_el.nsmap)

            param_name = find_attribute_value(param_el, data_type_ns, 'name')

            # FlowKit only supports linear, log, and logicle transformations in FlowJo WSP files.
            # All other bi-ex transforms implemented by FlowJo are undocumented and not reproducible
            if xform_type == 'linear':
                min_range = find_attribute_value(xform_el, transform_ns, 'minRange')
                max_range = find_attribute_value(xform_el, transform_ns, 'maxRange')
                xforms_lut[param_name] = _transforms.LinearTransform(
                    param_name,
                    param_t=float(max_range),
                    param_a=float(min_range)
                )
            elif xform_type == 'log':
                offset = find_attribute_value(xform_el, transform_ns, 'offset')
                decades = find_attribute_value(xform_el, transform_ns, 'decades')
                xforms_lut[param_name] = _wsp_transforms.WSPLogTransform(
                    param_name,
                    offset=float(offset),
                    decades=float(decades)
                )
            elif xform_type == 'logicle':
                # logicle transform has 4 parameters: T, W, M, and A
                # these are attributes of the 'logicle' element
                param_t = find_attribute_value(xform_el, transform_ns, 'T')
                param_w = find_attribute_value(xform_el, transform_ns, 'W')
                param_m = find_attribute_value(xform_el, transform_ns, 'M')
                param_a = find_attribute_value(xform_el, transform_ns, 'A')
                xforms_lut[param_name] = _transforms.LogicleTransform(
                    param_name,
                    param_t=float(param_t),
                    param_w=float(param_w),
                    param_m=float(param_m),
                    param_a=float(param_a)
                )
            elif xform_type == 'biex':
                # biex transform has 5 parameters, but only 2 are really used
                # these are attributes of the 'biex' element
                param_neg = find_attribute_value(xform_el, transform_ns, 'neg')
                param_width = find_attribute_value(xform_el, transform_ns, 'width')
                param_length = find_attribute_value(xform_el, transform_ns, 'length')
                param_max_range = find_attribute_value(xform_el, transform_ns, 'maxRange')
                param_pos = find_attribute_value(xform_el, transform_ns, 'pos')
                param_pos = round(float(param_pos), 2)

                if param_length != '256':
                    raise ValueError(
                        f"FlowJo biex 'length' parameter value of {param_length} is not supported."
                    )

                xforms_lut[param_name] = _wsp_transforms.WSPBiexTransform(
                    param_name,
                    negative=float(param_neg),
                    width=float(param_width),
                    positive=float(param_pos),
                    max_value=float(param_max_range)
                )
            elif xform_type == 'fasinh':
                # FlowJo's implementation of fasinh is slightly different from GML,
                # and uses an additional 'length' scale factor. However, this scaling
                # doesn't seem to affect the results, and we can use the regular
                # GML version of asinh. The xform_el also contains other
                # unnecessary parameters: 'length', 'maxRange', and 'W'
                param_t = find_attribute_value(xform_el, transform_ns, 'T')
                param_a = find_attribute_value(xform_el, transform_ns, 'A')
                param_m = find_attribute_value(xform_el, transform_ns, 'M')
                xforms_lut[param_name] = _transforms.AsinhTransform(
                    param_name,
                    param_t=float(param_t),
                    param_m=float(param_m),
                    param_a=float(param_a)
                )
            else:
                error_msg = f"FlowJo transform type '{xform_type}' is undocumented and not supported in FlowKit. "
                error_msg += "Please edit the workspace in FlowJo and save all channel transformations as either " \
                                "linear, log, biex, logicle, or ArcSinh"

                raise ValueError(error_msg)

        return xforms_lut

    def parse_wsp_groups(self,
                         group_node_list: list[etree._Element],
                         ns_map: dict,
                         gating_namespace: str,
                         data_type_namespace: str) -> dict:
        wsp_groups = {}
        for group_node in group_node_list:
            group_name: str = group_node.attrib["name"]

            group_node_group: etree._Element = group_node.find("Group", ns_map)
            if group_node_group is not None:
                sample_ids = self.parse_sampleIDs_from_sample_refs(group_node_group, ns_map)
            
            group_node_subpopulation = group_node.find("Subpopulations", ns_map)
            if group_node_subpopulation is not None:
                group_gates = self.parse_wsp_subpopulations(
                    group_node_subpopulation,
                    None,
                    gating_namespace,
                    data_type_namespace
                )
            else:
                group_gates = None
            
            wsp_groups[group_name] = {
                "gates": group_gates,
                "samples": sample_ids
            }

        return wsp_groups

    def parse_wsp_subpopulations(self,
                                 subpopulation: etree._Element,
                                 gate_path: Optional[list],
                                 gating_ns: str,
                                 data_type_ns: str) -> list[dict]:
        
        ns_map = subpopulation.nsmap

        gates = []
        
        populations: list[etree._Element] = subpopulation.findall("Population", ns_map)
        
        for population in populations:
            gate_path, parent_id = self.fetch_gate_path_and_parent(gate_path)
            gates.append(self.fetch_gate_from_xml_population(population,
                                                             ns_map,
                                                             gating_ns,
                                                             data_type_ns,
                                                             parent_id,
                                                             gate_path))

            subpopulations = population.findall("Subpopulations", ns_map)
            child_gate_path = gate_path.copy()
            child_gate_path.append(population.attrib["name"])
            for el in subpopulations:
                gates.extend(self.parse_wsp_subpopulations(el,
                                                           child_gate_path,
                                                           gating_ns,
                                                           data_type_ns))


        return gates

    def fetch_gate_from_xml_population(self,
                                       population: etree._Element,
                                       ns_map: dict,
                                       gating_ns: str,
                                       data_type_ns: str,
                                       parent_id: str,
                                       gate_path: list) -> dict:
        population_name = population.attrib["name"]
        owning_group = population.attrib["owningGroup"]
        
        gate_element = population.find("Gate", ns_map)
        gate_children: list[etree._Element] = gate_element.getchildren()

        gate_child_element = gate_children[0]
        # <Element {http://www.isac-net.org/std/Gating-ML/v2.0/gating}PolygonGate at 0x2206f425340>
        gate_type = gate_child_element.tag.partition("}")[-1]

        gate_class = wsp_gate_constructor_lut[gate_type]
        g: Union[GMLRectangleGate,
                 GMLPolygonGate,
                 WSPEllipsoidGate,
                 GMLQuadrantGate,
                 GMLBooleanGate] = gate_class(
                                       gate_child_element,
                                       gating_ns,
                                       data_type_ns
                                   )
        g.gate_name = population_name
        g.parent = parent_id

        return {'owning_group': owning_group,
                'gate': g,
                'gate_path': tuple(gate_path)}

    def fetch_gate_path_and_parent(self,
                                   gate_path: Optional[list]) -> tuple[list, str]:
        if gate_path is None:
            gate_path = ["root"]
            parent_id = None
        else:
            parent_id = gate_path[-1]

        return gate_path, parent_id

    def parse_sampleIDs_from_sample_refs(self,
                                         group_element: etree._Element,
                                         ns_map: dict) -> list[str]:
        sample_references: etree._Element = group_element.find("SampleRefs", ns_map)
        
        if sample_references is None:
            return []
        
        sample_reference_elements: list[etree._Element] = sample_references.findall("SampleRef", ns_map)
        return [sample_ref.attrib["sampleID"] for sample_ref in sample_reference_elements]
                  
        

    def parse_xml_group_nodes(self,
                              wsp_root: etree._Element,
                              ns_map: dict) -> list[etree._Element]:
        group_elements = self.parse_xml_groups(wsp_root, ns_map)
        return group_elements.findall("GroupNode", ns_map)

    def parse_xml_groups(self,
                         wsp_root: etree._Element,
                         ns_map: dict) -> etree._Element:
        return wsp_root.find("Groups", ns_map)

    def parse_xml_samples(self,
                          wsp_root: etree._Element,
                          ns_map: dict) -> list[etree._Element]:
        
        sample_list_elements = self.parse_xml_sample_list(wsp_root, ns_map)
        return sample_list_elements.findall("Sample", ns_map)

    def parse_xml_sample_list(self,
                              wsp_root: etree._Element,
                              ns_map: dict) -> etree._Element:
        return wsp_root.find("SampleList", ns_map)

    def _extract_namespaces(self,
                            file: str) -> tuple[etree._Element, str, str, str]:
        raw_wsp = etree.parse(file)
        wsp_root = raw_wsp.getroot()
        gating_ns = None
        data_type_ns = None
        transform_ns = None

        # find GatingML target namespace in the map
        for ns, url in wsp_root.nsmap.items():
            if url == 'http://www.isac-net.org/std/Gating-ML/v2.0/gating':
                gating_ns = ns
            elif url == 'http://www.isac-net.org/std/Gating-ML/v2.0/datatypes':
                data_type_ns = ns
            elif url == 'http://www.isac-net.org/std/Gating-ML/v2.0/transformations':
                transform_ns = ns

        return wsp_root, gating_ns, data_type_ns, transform_ns



class DivaWorkspace:
    """
    Class to represent a diva workspace.
    Built to be compatible with the flowkit package.
    Q1-Q4 Quadrant Gates are parsed as Polygon-Gates in contrast to GMLRectangle in Flowkit
    Quadrant Gates have an upper limit here in contrast to FlowKit were "None" is passed to extend the gates to infinity
    Binner-Regions are ignored as they define the points where the quadrant gates lie
    Code is inspired by CytoML, however one notable change: "restore to logicle scale" was removed
    TODO: Q1-1 is not supported currently...
    TODO: Add gate names for Q-gates -> check with Diva which Qs are which gates
    TODO: refactor parse diva gate coordinates
    """
    def __init__(self,
                 file: str,):

        if not self._correct_suffix(file):
            raise ValueError("Only XML Diva Workspaces are supported")

        self.wsp_dict = self.create_workspace_dictionary(file)

    def __repr__(self):
        
        return (
            f"{self.__class__.__name__}(" +
            f"{len(list(self.wsp_dict.keys()))} groups: {list(self.wsp_dict.keys())}, " +
            f"{len(list(self.wsp_dict['All Samples'].keys()))} entries.)"
        )

    def parse_experiment(self,
                         raw_wsp: etree._ElementTree) -> etree._Element:
        root: etree._Element = raw_wsp.getroot()
        experiment_list = root.getchildren()
        if len(experiment_list) != 1:
            raise ValueError("More than one experiment detected...???")
        return experiment_list[0]
    
    def parse_tubes(self,
                    raw_wsp: etree._ElementTree) -> list[etree._Element]:
        experiment = self.parse_experiment(raw_wsp)
        log_decades = int(experiment.find("log_decades").text)
        if log_decades == 4:
            min_val = 26
        elif log_decades == 5:
            min_val = 2.6
        else:
            raise ValueError("Decade is neither 4 or 5...")
        specimens = experiment.findall("specimen")
        return list(itertools.chain(*[specimen.findall("tube") for specimen in specimens]))
    
    def create_workspace_dictionary(self,
                                    file: str) -> dict:
        
        raw_wsp = self.parse_raw_data(file)
        tubes = self.parse_tubes(raw_wsp)

        #self.version = dict(root.items())["version"]
        #self.experiment_name = dict(experiment.items())["name"]

        ### create custom "All Samples" key to be compatible with Dataset Class
        wsp_dict = {"All Samples": {}}

        for tube in tubes:
            tube_id = tube.find("data_filename").text
            if "Compensation Control" in tube_id:
                print(f"warning... currently not implemented, skipping tube {tube_id}")
                continue
            wsp_dict["All Samples"][tube_id] = {}
            transform_lut = self._parse_diva_transformation(instrument_settings = tube.find("instrument_settings"))
            wsp_dict["All Samples"][tube_id]["transformations"] = transform_lut
            wsp_dict["All Samples"][tube_id]["transforms"] = [xform for (_, xform) in transform_lut.items()]
            wsp_dict["All Samples"][tube_id]["gates"] = self._parse_diva_gates(gate_elements = tube.find("gates"),
                                                                               transform_dict = transform_lut)
            wsp_dict["All Samples"][tube_id]["compensation"] = self._parse_diva_compensation(instrument_settings = tube.find("instrument_settings"))


        return wsp_dict

    def parse_raw_data(self,
                       file: str) -> etree._Element:
        return etree.parse(file)

    def _parse_diva_transformation(self,
                                   instrument_settings) -> dict:
        transform_dict = {}
        if instrument_settings is None:
            return {}
        use_auto_biexp_scale = bool(instrument_settings.find("use_auto_biexp_scale").text)
        biexp_scale_node = "comp_biexp_scale" if use_auto_biexp_scale else "manual_biexp_scale"
        parameters = instrument_settings.findall("parameter")
        #scale_settings = {}
        for param in parameters:
            param_name = param.attrib["name"]
            #scale_settings[param] = {}
            is_log = param.find("is_log").text == "true"
            #scale_settings[param]["scale_minimum"] = float(param.find("min"))
            scale_minimum = float(param.find("min").text)
            #scale_settings[param]["scale_maximum"] = float(param.find("max"))
            scale_maximum = float(param.find("max").text)
            #scale_settings[param]["biexp_scale"] = abs(float(param.find(biexp_scale_node).text))

            if is_log:
                biexp_scale = abs(float(param.find(biexp_scale_node).text))
                if biexp_scale == 0:
                    transform_dict[param_name] = _transforms.LogTransform(
                                                    transform_id = param_name,
                                                    param_t = 10**scale_maximum,
                                                    param_m = 4.5
                                                )
                else:
                    w = max(0, (4.5 - np.log10(10**scale_maximum/biexp_scale)) * 0.5) ## source: CytoML; automatically sets to 0 if less than 0
                    transform_dict[param_name] = _transforms.LogicleTransform(
                                                    transform_id = param_name,
                                                    param_w = w,
                                                    param_t = 10**scale_maximum,
                                                    param_m = 4.5,
                                                    param_a = 0 ## leave at 0 for now, cytoML and FlowJo do so...
                                                )
            else:
                transform_dict[param_name] = _transforms.LinearTransform(
                                                transform_id = param_name,
                                                param_t = scale_maximum,
                                                param_a = 0
                                            )              

        return transform_dict


    def _parse_diva_compensation(self,
                                 instrument_settings: etree._Element) -> Matrix:
        parameters = instrument_settings.findall("parameter")
        fluorochrome_list = [dict(param.items())["name"] for param in parameters if
                             param.find("can_be_compensated") is not None and
                             param.find("can_be_compensated").text == "true"]
        comp_matrix = np.zeros((len(fluorochrome_list),len(fluorochrome_list)))
        for i, channel in enumerate(fluorochrome_list):
            comp_list = [param.find("compensation").getchildren() for param in parameters if dict(param.items())["name"] == channel]
            comp_matrix[:, i] = [value.text for value in comp_list[0]]
        comp_matrix = np.linalg.solve(comp_matrix, np.eye(N = comp_matrix.shape[0], M = comp_matrix.shape[1]))
        return Matrix(matrix_id = "Acquisition Defined",
                      detectors = fluorochrome_list,
                      fluorochromes = fluorochrome_list,
                      spill_data_or_file = comp_matrix)

    def _parse_diva_gates(self,
                          gate_elements: etree._Element,
                          transform_dict: dict) -> list:
        
        gate_list = []
        for gate in gate_elements.getchildren():
            gate_name = gate.find("name").text
            gate_path = gate.attrib["fullname"].split("\\")
            gate_path.remove(gate_name)
            if gate_path == []:
                gate_path.append("root")
            for i, el in enumerate(gate_path):
                if el == "All Events":
                    gate_path[i] = "root"
                    break
            gate_path = tuple(gate_path)
            if gate.find("region") is None:
                assert "All Events" or "Rest" in gate_name, f"some other gate than all events encountered... {gate_name}"
            else:
                x_param, y_param, gate_type, gate_coordinates = self._process_gate_coordinates(gate, transform_dict)

                dims = []
                for i, param in enumerate([x_param, y_param]):
                    if param is not None:
                        new_dim = Dimension(dimension_id = param,
                                            compensation_ref = None,
                                            transformation_ref = param,
                                            range_min = np.min(gate_coordinates[:,i]),
                                            range_max = np.max(gate_coordinates[:,i])
                                            )
                        dims.append(new_dim)
                if gate_type == "POLYGON_REGION":
                    gate_list.append({"gate": PolygonGate(gate_name, dims, gate_coordinates), "gate_path": gate_path})
                elif gate_type in ["RECTANGLE_REGION", "INTERVAL_REGION"]:
                    gate_list.append({"gate": RectangleGate(gate_name, dims), "gate_path": gate_path})
                else:
                    print(gate_type)

        return gate_list

    def _process_gate_coordinates(self,
                                  gate: etree._Element,
                                  transform_lut: dict) -> np.ndarray:
        
        
        gate_region = dict(gate.find("region").items())
        gate_type = gate_region["type"]

        dim_counter = 0
        try:
            x_param = gate_region["xparm"]
            dim_counter += 1
        except KeyError:
            x_param = None

        try:
            y_param = gate_region["yparm"]
            dim_counter += 1
        except KeyError:
            y_param = None
            assert gate_type == "INTERVAL_REGION"

        gate_points = gate.find("region").find("points").findall("point")
        gate_points = pd.DataFrame([dict(point.items()) for point in gate_points]).to_numpy(dtype = np.float64)
        is_x_scaled = gate.find("is_x_parameter_scaled").text == "true"
        is_y_scaled = gate.find("is_y_parameter_scaled").text == "true"
        x_parameter_scale_value = float(gate.find("x_parameter_scale_value").text)
        y_parameter_scale_value = float(gate.find("y_parameter_scale_value").text)

        # sourcery skip: extract-duplicate-method, merge-else-if-into-elif
        if is_x_scaled:
            gate_points[:,0] = gate_points[:,0] / 4096
            if gate.find("is_x_parameter_log").text == "true":
                # create temporary transformation, seems to be always a logicle transform so no need for log transform for now
                # TODO: check if implementing log-transform for no apparent scale value makes sense
                scale_value = x_parameter_scale_value
                w = max(0, (4.5 - np.log10(10**5.4185380935668945/scale_value)) * 0.5) ## source: CytoML; automatically sets to 0 if less than 0
                temp_transform = _transforms.LogicleTransform(
                                                transform_id = "temporary",
                                                param_w = w,
                                                param_t = 10**5.4185380935668945,
                                                param_m = 4.5,
                                                param_a = 0 ## leave at 0 for now, cytoML and FlowJo do so...
                                            )
                # CytoML does multiply with 4.5 to "restore to the logicle scale"
                # Comparison with flowkit seems to indicate that this step is not necessary
                # gate_points[:,0] = gate_points[:,0] * 4.5 # restore it to the logicle scale
                gate_points[:,0] = temp_transform.inverse(gate_points[:,0])
            else:
                gate_points[:,0] = gate_points[:,0] * 10**5.4185380935668945

        else: ## x is not scaled
            if gate.find("is_x_parameter_log").text == "true":
                gate_points[:,0] = 10 ** gate_points[:,0]
            ## implicit code chunk
            #else:
            #    gate_points[:,0] = gate_points[:,0]

        ## transform them all for compatibility with FlowKit
        if x_param is not None:
            gate_points[:,0] = transform_lut[x_param].apply(gate_points[:,0])


        if is_y_scaled:
            gate_points[:,1] = gate_points[:,1] / 4096
            if gate.find("is_y_parameter_log").text == "true":
                # create temporary transformation, seems to be always a logicle transform so no need for log transform for now
                # TODO: check if implementing log-transform for no apparent scale value makes sense
                scale_value = y_parameter_scale_value
                w = max(0, (4.5 - np.log10(10**5.4185380935668945/scale_value)) * 0.5) ## source: CytoML; automatically sets to 0 if less than 0
                temp_transform = _transforms.LogicleTransform(
                                                transform_id = "temporary",
                                                param_w = w,
                                                param_t = 10**5.4185380935668945,
                                                param_m = 4.5,
                                                param_a = 0 ## leave at 0 for now, cytoML and FlowJo do so...
                                            )
                # CytoML does multiply with 4.5 to "restore to the logicle scale"
                # Comparison with flowkit seems to indicate that this step is not necessary
                # gate_points[:,1] = gate_points[:,1] * 4.5 # restore it to the logicle scale

                gate_points[:,1] = temp_transform.inverse(gate_points[:,1])
            else:
                gate_points[:,1] = gate_points[:,1] * 10**5.4185380935668945

        else: ## y is not scaled
            if gate.find("is_y_parameter_log").text == "true":
                gate_points[:,1] = 10 ** gate_points[:,1]
            ## implicit code chunk
            #else:
            #    gate_points[:,1] = gate_points[:,1]

        if y_param is not None:
            gate_points[:,1] = transform_lut[y_param].apply(gate_points[:,1])
        
        return x_param, y_param, gate_type, gate_points


    def _correct_suffix(self,
                        file_name: str) -> bool:
        return file_name.endswith(".xml")