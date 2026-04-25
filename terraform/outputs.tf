output "public_ip" {
  description = "Static Elastic IP attached to the instance."
  value       = aws_eip.clyde.public_ip
}

output "public_dns" {
  description = "AWS-assigned public DNS for the instance."
  value       = aws_instance.clyde.public_dns
}

output "url" {
  description = "Clyde web URL (HTTPS if domain set, HTTP via EIP otherwise)."
  value       = var.domain != "" ? "https://${var.domain}" : "http://${aws_eip.clyde.public_ip}"
}

output "ssh_command" {
  description = "SSH command (assumes the matching private key is present)."
  value       = "ssh ubuntu@${aws_eip.clyde.public_ip}"
}

output "tail_logs" {
  description = "How to watch the cloud-init bootstrap from your laptop."
  value       = "ssh ubuntu@${aws_eip.clyde.public_ip} 'sudo tail -f /var/log/cloud-init-output.log'"
}
