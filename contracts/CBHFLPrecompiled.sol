pragma solidity ^0.4.24;

contract CommitteePrecompiled {
    function RegisterNode(string role) public; //节点注册

    function QueryCurrentEpoch() public view returns (int256); //获取当前epoch

    function QueryGlobalProtos() public view returns (string, int); //获取全局Protos

    function UploadLocalProtos(string protos, int256 epoch) public; //上传本地Protos

    function QueryProtosUpdates() public view returns (string); //获取所有本地Protos更新

    function UpdateGlobalProtos(string protos, int256 epoch) public; //更新全局Protos
}
